"""Per-call token / latency / cost accounting ("no evaluation run without cost logging")."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# USD per 1M tokens (input, output). Local models cost $0; every call is also priced at gpt-4o-mini
# rates ("API-equivalent") so configurations remain comparable with hosted deployments.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (1.00, 5.00),
    "claude-sonnet-5-5": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}
REFERENCE_MODEL = "gpt-4o-mini"


@dataclass
class CallRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    usd: float
    usd_equiv: float


class CostTracker:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int, latency_s: float) -> CallRecord:
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        r_in, r_out = PRICES[REFERENCE_MODEL]
        rec = CallRecord(agent, model, input_tokens, output_tokens, round(latency_s, 3),
                         (input_tokens * p_in + output_tokens * p_out) / 1e6,
                         (input_tokens * r_in + output_tokens * r_out) / 1e6)
        self.records.append(rec)
        return rec

    def summary(self) -> dict:
        by_agent: dict[str, dict] = {}
        for r in self.records:
            a = by_agent.setdefault(r.agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0})
            a["calls"] += 1
            a["input_tokens"] += r.input_tokens
            a["output_tokens"] += r.output_tokens
            a["latency_s"] = round(a["latency_s"] + r.latency_s, 3)
        return {
            "calls": len(self.records),
            "input_tokens": sum(r.input_tokens for r in self.records),
            "output_tokens": sum(r.output_tokens for r in self.records),
            "llm_latency_s": round(sum(r.latency_s for r in self.records), 3),
            "usd": round(sum(r.usd for r in self.records), 6),
            "usd_equiv": round(sum(r.usd_equiv for r in self.records), 6),
            "by_agent": by_agent,
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({"summary": self.summary(), "calls": [asdict(r) for r in self.records]},
                                   indent=2), encoding="utf-8")
