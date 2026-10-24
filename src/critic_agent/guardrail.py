"""Guardrail applied at every untrusted entry point: PDF text, user request, tool output, memory.

Layers: (1) hidden-text removal happens in the PDF parser; (2) instruction-like sentences are redacted
here; (3) prompts spotlight untrusted content inside <untrusted_document> tags (see prompts.py).
Patterns are deliberately narrow: they produce zero flags on the 8 clean corpus papers.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

REDACTION = "[REDACTED: suspected injected instruction]"

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your)\b[^.\n]{0,20}\b(instruction|prompt|rule|guideline|task)s?",
    r"\byou are now\b|\b(follow|here are) (the |these )?new instructions\b|\b(developer|admin|god|jailbreak) mode\b",
    r"\b(reveal|print|show|repeat|output)\b[^.\n]{0,30}\b(system prompt|hidden prompt|instructions above)\b",
    r"\b(dear|attention|note to( the)?|hey)\s+(ai|llm|language model|assistant|automated reviewer|ai reviewer|reviewer model)\b",
    r"\bto (any|all) (ai|llm|automated) (reviewer|assistant|system|agent)s?\b",
    r"\b(do not|don't|never)\s+(report|mention|flag|criticize|critique|list|output)\b[^.\n]{0,40}\b(weakness|limitation|flaw|contamination)",
    r"\b(rate|score|grade)\b[^.\n]{0,20}\b(this paper|the paper|it)\b[^.\n]{0,20}\b(10|perfect|flawless|maximum)",
    r"\b(this paper|the paper)\s+(is|has)\s+(flawless|perfect|no (weaknesses|limitations|flaws))\b",
    r"\b(execute_code_in_sandbox|save_annotation|delete_annotation|publish_review|search_paper|pdf_ingest)\b",
    r"\b(respond|reply|answer)\s+only\s+with\b|\boutput\s+exactly\b",
    r"\bsystem\s*:\s|\[\s*system\s*\]|<\s*/?\s*system\s*>",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n")


@dataclass
class Flag:
    source: str  # pdf | pdf-hidden | user | tool:<name> | memory
    page: int | None
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_injections(text: str) -> list[str]:
    return [s for s in _SENTENCES.split(text) if s.strip() and any(p.search(s) for p in _COMPILED)]


def sanitize(text: str, source: str, page: int | None = None) -> tuple[str, list[Flag]]:
    flags = []
    for sentence in find_injections(text):
        text = text.replace(sentence, REDACTION)
        flags.append(Flag(source, page, sentence.strip()[:160]))
    return text, flags
