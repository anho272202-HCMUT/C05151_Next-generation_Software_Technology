"""Long-term (episodic) memory: lessons learned from the Evaluator's verdicts, stored in ChromaDB.

Write policy — after a run, per axis, only if the final attempt was ACCEPTED (grounded, supported, score
above threshold):
  * "fix" lesson when earlier attempts on that axis were rejected: what was wrong + what was accepted;
  * "pattern" lesson otherwise: the accepted critique as an example.
Text is sanitised before it is stored and again when it is read; duplicates are skipped (content hash).
Retrieve policy — top-k (3) by similarity to the current paper card, EXCLUDING lessons that came from the
same paper (no answer leakage), injected into prompts as examples inside <untrusted> tags.
`forget(paper_id)` removes everything a paper contributed (recovery from memory poisoning).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import chromadb

from .config import Settings
from .guardrail import sanitize
from .retrieval import Embedder, HashEmbedder, OllamaEmbedder

COLLECTION = "critique_lessons"


class LessonStore:
    def __init__(self, path: Path, embedder: Embedder, k: int = 3):
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        # One collection per embedder: vectors of different models/dimensions must never mix.
        self._col = client.get_or_create_collection(f"{COLLECTION}_{embedder.name}", metadata={"hnsw:space": "cosine"})
        self._embedder = embedder
        self._k = k

    @classmethod
    def from_settings(cls, settings: Settings) -> "LessonStore":
        embedder = (HashEmbedder() if settings.llm_provider == "mock"
                    else OllamaEmbedder(settings.ollama_base_url, settings.embed_model))
        return cls(settings.memory_dir, embedder)

    def write_from_run(self, paper_id: str, attempts: list[dict]) -> int:
        by_axis: dict[str, list[dict]] = defaultdict(list)
        for a in sorted(attempts, key=lambda a: a["round"]):
            by_axis[a["axis"]].append(a)
        written = 0
        for axis, tries in by_axis.items():
            final = tries[-1]
            if not final.get("accepted"):
                continue
            rejected = [t for t in tries[:-1] if not t.get("accepted")]
            if rejected:
                text = (f"[{axis}] A first critique was rejected because: {rejected[0].get('feedback', '')[:200]} "
                        f"The accepted critique was: {final['weakness'][:300]}")
                kind = "fix"
            else:
                text = f"[{axis}] Accepted critique example: {final['weakness'][:300]}"
                kind = "pattern"
            written += self._add(text, paper_id, axis, kind, float(final.get("score", 0)))
        return written

    def _add(self, text: str, paper_id: str, axis: str, kind: str, score: float) -> int:
        text, _ = sanitize(text, "memory")
        doc_id = hashlib.sha1(text.encode()).hexdigest()[:16]
        if self._col.get(ids=[doc_id])["ids"]:
            return 0
        self._col.add(ids=[doc_id], documents=[text], embeddings=[self._embedder.embed([text])[0].tolist()],
                      metadatas=[{"paper_id": paper_id, "axis": axis, "kind": kind, "score": score}])
        return 1

    def retrieve(self, query: str, paper_id: str) -> list[str]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[self._embedder.embed([query])[0].tolist()],
                              n_results=min(self._k * 3, self._col.count()), where={"paper_id": {"$ne": paper_id}})
        docs = (res.get("documents") or [[]])[0][: self._k]
        return [sanitize(d, "memory")[0] for d in docs]

    def forget(self, paper_id: str) -> int:
        ids = self._col.get(where={"paper_id": paper_id})["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._col.count()

    def dump(self) -> list[dict]:
        got = self._col.get()
        return [{"id": i, "text": d, **m} for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]
