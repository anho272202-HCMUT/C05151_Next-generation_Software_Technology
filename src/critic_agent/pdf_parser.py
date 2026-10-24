"""PDF parsing into page-numbered sentences, with detection of visually hidden text.

Sentences get stable ids `p<page>s<n>`; critics cite ids and the system copies the verbatim text,
so a citation can never be a paraphrase.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

HIDDEN_COLORS = {0xFFFFFF, 0xFEFEFE, 0xFDFDFD}
TINY_FONT_PT = 2.0
# Short white spans are usually figure labels on dark fills; injected payloads are sentences.
HIDDEN_MIN_WORDS = 6
MAX_SENTENCE_WORDS = 60
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[“\"])")


@dataclass
class HiddenSpan:
    page: int
    text: str
    reason: str  # white-text | tiny-font


@dataclass
class Sentence:
    sid: str
    page: int
    text: str


@dataclass
class ParsedPaper:
    paper_id: str
    path: str
    title: str
    num_pages: int
    raw_pages: list[str]  # everything a naive extractor sees (used when the guardrail is off)
    visible_pages: list[str]  # hidden payload blocks removed
    hidden_spans: list[HiddenSpan] = field(default_factory=list)


def paper_id_for(path: Path) -> str:
    return f"{path.stem}-{hashlib.sha1(path.read_bytes()).hexdigest()[:8]}"


def parse_pdf(path: Path) -> ParsedPaper:
    doc = pymupdf.open(path)
    raw, visible, hidden = [], [], []
    for page_no, page in enumerate(doc, start=1):
        raw.append(page.get_text("text"))
        text, spans = _split_visible(page, page_no)
        visible.append(text)
        hidden.extend(spans)
    title = (doc.metadata or {}).get("title") or _guess_title(raw[0] if raw else "")
    return ParsedPaper(paper_id_for(path), str(path), title.strip(), doc.page_count, raw, visible, hidden)


def split_sentences(pages: list[str]) -> list[Sentence]:
    out: list[Sentence] = []
    for page_no, text in enumerate(pages, start=1):
        text = re.sub(r"-\s*\n\s*", "", text)
        text = " ".join(text.split())
        n = 0
        for chunk in _SPLIT.split(text):
            words = chunk.split()
            for start in range(0, len(words), MAX_SENTENCE_WORDS):
                piece = " ".join(words[start : start + MAX_SENTENCE_WORDS])
                if len(piece) >= 25:
                    out.append(Sentence(f"p{page_no}s{n}", page_no, piece))
                    n += 1
    return out


def _split_visible(page: pymupdf.Page, page_no: int) -> tuple[str, list[HiddenSpan]]:
    """Rebuild page text from what a human can see.

    A block counts as a hidden payload when one of its hidden spans has >= HIDDEN_MIN_WORDS words; then
    every hidden span of that block is dropped (a wrapped payload ends in short continuation lines).
    """
    lines: list[str] = []
    found: list[HiddenSpan] = []
    for block in page.get_text("dict").get("blocks", []):
        spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
        hidden = [s for s in spans if _hidden_reason(s)]
        drop = any(len(s.get("text", "").split()) >= HIDDEN_MIN_WORDS for s in hidden)
        if drop:
            found.append(HiddenSpan(page_no, " ".join(s["text"].strip() for s in hidden), _hidden_reason(hidden[0])))
        for line in block.get("lines", []):
            lines.append("".join(s.get("text", "") for s in line.get("spans", [])
                                 if not (drop and _hidden_reason(s))))
        lines.append("")
    return "\n".join(lines), found


def _hidden_reason(span: dict) -> str | None:
    if span.get("color") in HIDDEN_COLORS:
        return "white-text"
    if span.get("size", 10) < TINY_FONT_PT:
        return "tiny-font"
    return None


def _guess_title(first_page: str) -> str:
    for line in first_page.splitlines():
        line = line.strip()
        if len(line) > 12 and not line.lower().startswith(("published", "arxiv", "preprint")):
            return line
    return "Untitled"
