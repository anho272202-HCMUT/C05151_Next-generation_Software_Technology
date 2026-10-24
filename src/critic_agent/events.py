"""Run trace: an ordered event log that is saved as trace.jsonl and replayed by the web Explorer.

Event types: run_start, node_start, node_end, llm_call, tool_call, tool_denied, guardrail_flag,
critique, evaluation, reflection, run_end.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Listener = Callable[[dict], None]


class Trace:
    def __init__(self, run_id: str, listener: Listener | None = None):
        self.run_id = run_id
        self.events: list[dict] = []
        self._t0 = time.perf_counter()
        self._listener = listener

    def emit(self, type_: str, agent: str = "", **data: Any) -> dict:
        event = {"seq": len(self.events), "t": round(time.perf_counter() - self._t0, 3), "type": type_,
                 "agent": agent, **data}
        self.events.append(event)
        if self._listener is not None:
            self._listener(event)
        return event

    def save(self, path: Path) -> None:
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.events) + "\n", encoding="utf-8")
