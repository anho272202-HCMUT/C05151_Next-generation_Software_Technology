"""Per-paper hybrid retrieval: sentence windows indexed with BM25 and dense embeddings, fused by RRF."""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from .pdf_parser import Sentence

WINDOW = 4  # sentences per chunk
STRIDE = 3
RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    page: int
    sentences: list[Sentence]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


def build_chunks(sentences: list[Sentence]) -> list[Chunk]:
    by_page: dict[int, list[Sentence]] = {}
    for s in sentences:
        by_page.setdefault(s.page, []).append(s)
    chunks = []
    for page, sents in by_page.items():
        for n, start in enumerate(range(0, len(sents), STRIDE)):
            window = sents[start : start + WINDOW]
            chunks.append(Chunk(f"p{page}c{n}", page, window))
            if start + WINDOW >= len(sents):
                break
    return chunks


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic bag-of-words embedder (offline tests / mock provider)."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                arr[row, zlib.crc32(tok.encode()) % self._dim] += 1.0
        return _normalise(arr)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.name = model.replace(":", "_")
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):
            resp = httpx.post(self._url, json={"model": self._model, "input": texts[i : i + 64]}, timeout=300)
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return _normalise(np.asarray(vectors, dtype=np.float32))


def _normalise(arr: np.ndarray) -> np.ndarray:
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


class PaperIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, cache_file: Path | None = None):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._embedder = embedder
        self._dense = self._load_or_embed(cache_file) if embedder else None

    def _load_or_embed(self, cache_file: Path | None) -> np.ndarray:
        if cache_file and cache_file.exists():
            cached = np.load(cache_file)
            if cached.shape[0] == len(self.chunks):
                return cached
        # nomic-embed-text expects task prefixes; harmless for other embedders.
        dense = self._embedder.embed([f"search_document: {c.text}" for c in self.chunks])
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, dense)
        return dense

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(query))
        fused = {int(i): 1.0 / (RRF_K + r) for r, i in enumerate(np.argsort(-scores)[:50])}
        if self._dense is not None:
            q = self._embedder.embed([f"search_query: {query}"])[0]
            for r, i in enumerate(np.argsort(-(self._dense @ q))[:50]):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + r)
        return [self.chunks[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]
