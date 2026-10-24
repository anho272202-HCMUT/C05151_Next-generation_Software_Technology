"""Short-term memory: the per-critic evidence window.

Each critic invocation fills a fresh window with retrieved sentences in rank order (de-duplicated);
whatever does not fit the token budget is dropped from the tail, so the prompt always fits the model's
context. The window's ids are also the only ids a critic may cite. Long-term memory is in lessons.py.
"""

from __future__ import annotations

from .llm import approx_tokens


class EvidenceWindow:
    def __init__(self, token_budget: int):
        self._budget = token_budget
        self._items: dict[str, str] = {}
        self._used = 0
        self.dropped = 0

    def add(self, sid: str, text: str) -> bool:
        if sid in self._items:
            return False
        cost = approx_tokens(text) + 4
        if self._used + cost > self._budget:
            self.dropped += 1
            return False
        self._items[sid] = text
        self._used += cost
        return True

    @property
    def ids(self) -> set[str]:
        return set(self._items)

    @property
    def tokens(self) -> int:
        return self._used

    def render(self) -> str:
        return "\n".join(f"[{sid}] {text}" for sid, text in self._items.items())

    def __len__(self) -> int:
        return len(self._items)
