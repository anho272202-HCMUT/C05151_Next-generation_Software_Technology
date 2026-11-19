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

TINY_FONT_PT = 2.5  # scaled-down figure text in real papers is ~3 pt; a payload nobody can read is below this
MIN_CONTRAST = 1.5  # WCAG contrast ratio below which text is practically invisible (white on white is 1.0)
# Short hidden spans are usually figure labels on dark fills; injected payloads are sentences.
HIDDEN_MIN_WORDS = 6
MAX_SENTENCE_WORDS = 60
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[“\"])")


@dataclass
class HiddenSpan:
    page: int
    text: str
    reason: str  # low-contrast | tiny-font | covered | transparent


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


def _luminance(color) -> float:
    rgb = tuple(color) if color is not None else (0.0, 0.0, 0.0)
    if len(rgb) == 1:
        rgb = rgb * 3
    elif len(rgb) == 4:  # CMYK
        c, m, y, k = rgb
        rgb = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(fg, bg) -> float:
    hi, lo = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _hidden_regions(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Bounding boxes of text runs that a human cannot read, with the reason.

    Uses the paint order of text runs and filled shapes: the background of a run is the last opaque shape drawn
    under it (white page otherwise); a shape drawn *after* the run that covers it hides it.
    """
    # plain opaque rectangles only: frames and clipped figure backgrounds also appear as filled paths
    fills = [d for d in page.get_drawings() if d.get("fill") is not None and (d.get("fill_opacity") or 1) >= 0.95
             and d["rect"].width > 2 and d["rect"].height > 2 and [i[0] for i in d["items"]] == ["re"]]
    page_area = page.rect.get_area()
    regions = []
    for run in page.get_texttrace():
        box = pymupdf.Rect(run["bbox"])
        if box.is_empty or not run.get("chars"):
            continue
        under = [d for d in fills if d["seqno"] < run["seqno"] and d["rect"].contains(box.tl + (box.br - box.tl) * 0.5)]
        background = under[-1]["fill"] if under else (1, 1, 1)
        if (run.get("opacity") or 1) < 0.1:
            reason = "transparent"
        elif run["size"] < TINY_FONT_PT:
            reason = "tiny-font"
        elif _contrast(run["color"], background) < MIN_CONTRAST:
            reason = "low-contrast"
        elif any(d["seqno"] > run["seqno"] and d["rect"].contains(box) and d["rect"].get_area() < 0.5 * page_area
                 and page.rect.contains(d["rect"] + (-2, -2, 2, 2)) for d in fills):
            reason = "covered"
        else:
            continue
        regions.append((box, reason))
    return regions


def _split_visible(page: pymupdf.Page, page_no: int) -> tuple[str, list[HiddenSpan]]:
    """Rebuild page text from what a human can see.

    A block counts as a hidden payload when one of its hidden spans has >= HIDDEN_MIN_WORDS words; then
    every hidden span of that block is dropped (a wrapped payload ends in short continuation lines).
    """
    regions = _hidden_regions(page)

    def reason_of(span: dict) -> str | None:
        center = pymupdf.Rect(span["bbox"]).tl + (pymupdf.Rect(span["bbox"]).br - pymupdf.Rect(span["bbox"]).tl) * 0.5
        return next((why for box, why in regions if box.contains(center)), None)

    lines: list[str] = []
    found: list[HiddenSpan] = []
    for block in page.get_text("dict").get("blocks", []):
        spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
        hidden = [s for s in spans if reason_of(s)]
        drop = any(len(s.get("text", "").split()) >= HIDDEN_MIN_WORDS for s in hidden)
        if drop:
            found.append(HiddenSpan(page_no, " ".join(s["text"].strip() for s in hidden), reason_of(hidden[0])))
        for line in block.get("lines", []):
            lines.append("".join(s.get("text", "") for s in line.get("spans", [])
                                 if not (drop and reason_of(s))))
        lines.append("")
    return "\n".join(lines), found


def _guess_title(first_page: str) -> str:
    for line in first_page.splitlines():
        line = line.strip()
        if len(line) > 12 and not line.lower().startswith(("published", "arxiv", "preprint")):
            return line
    return "Untitled"
