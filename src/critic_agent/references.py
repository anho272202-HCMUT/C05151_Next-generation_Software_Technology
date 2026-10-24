"""Reference-list extraction and source classification ("which sources does this paper rely on?").

Heuristic by design: it splits the References section into entries, then classifies each entry as
peer-reviewed venue, arXiv/preprint, web/software, or other, and extracts the year. The resulting
profile backs the pitch's "unchecked citations" problem: a paper whose evidence rests mostly on
preprints or blog posts deserves a sharper question.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

_HEADING = re.compile(r"(?m)^\s*(?:\d+\s+)?(References|REFERENCES|Bibliography)\s*$")
_END = re.compile(r"(?m)^\s*(?:[A-H]\s+[A-Z][A-Za-z ]{3,40}|Appendix|APPENDIX|Supplementary Material|NeurIPS Paper "
                  r"Checklist|Checklist)\s*$")
_BRACKET = re.compile(r"(?:^|\s)\[(\d{1,3})\]\s")
# An author-year entry ends with "...2023." or an arXiv id "...2204.01691."; the next one starts uppercase.
_AUTHOR_YEAR_SPLIT = re.compile(r"(?:(?<=\d{4}\.)|(?<=\d{4}[a-z]\.))\s+(?=[A-Z])")
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
PEER_REVIEWED = re.compile(
    r"Proceedings|Conference|Symposium|Workshop|Journal|Transactions|NeurIPS|NIPS|ICLR|ICML|ACL|EMNLP|NAACL|"
    r"COLING|ICSE|FSE|ESEC|ISSTA|ASE\b|TOSEM|TSE\b|OOPSLA|PLDI|AAAI|IJCAI|CVPR|ICCV|KDD|COLM|Nature|Science",
    re.IGNORECASE)
PREPRINT = re.compile(r"arXiv|preprint|CoRR|openreview", re.IGNORECASE)
WEB = re.compile(r"https?://|github\.com|Retrieved|blog|technical report|tech report", re.IGNORECASE)


@dataclass
class Reference:
    text: str
    kind: str  # peer-reviewed | preprint | web | other
    year: int | None


def reference_text(doc: pymupdf.Document, max_pages: int = 10) -> str:
    """Text of the References section (from its heading to the appendix/checklist or end)."""
    for i in range(2, doc.page_count):
        page_text = doc[i].get_text()
        match = _HEADING.search(page_text)
        if not match:
            continue
        chunks = [page_text[match.end():]]
        for j in range(i + 1, min(doc.page_count, i + max_pages)):
            nxt = doc[j].get_text()
            end = _END.search(nxt)
            if end:
                chunks.append(nxt[: end.start()])
                break
            chunks.append(nxt)
        return " ".join(" ".join(chunks).replace("-\n", "").split())
    return ""


def split_entries(text: str) -> list[str]:
    if len(_BRACKET.findall(text)) >= 10:
        parts = _BRACKET.split(" " + text)
        entries = [parts[k] for k in range(2, len(parts), 2)]
    else:
        entries = _AUTHOR_YEAR_SPLIT.split(text)
    return [e.strip() for e in entries if len(e.strip()) > 25]


def classify(entry: str) -> Reference:
    years = [int(y) for y in _YEAR.findall(entry)]
    if PEER_REVIEWED.search(entry) and not re.search(r"arXiv preprint", entry, re.IGNORECASE):
        kind = "peer-reviewed"
    elif PREPRINT.search(entry):
        kind = "preprint"
    elif WEB.search(entry):
        kind = "web"
    else:
        kind = "other"
    return Reference(entry[:300], kind, years[-1] if years else None)


def reference_profile(refs: list[Reference]) -> dict:
    kinds = Counter(r.kind for r in refs)
    years = [r.year for r in refs if r.year]
    total = len(refs) or 1
    return {
        "total": len(refs),
        "peer_reviewed": kinds["peer-reviewed"], "preprint": kinds["preprint"], "web": kinds["web"],
        "other": kinds["other"],
        "peer_reviewed_share": round(kinds["peer-reviewed"] / total, 2),
        "median_year": sorted(years)[len(years) // 2] if years else None,
    }
