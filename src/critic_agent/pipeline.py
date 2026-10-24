"""LangGraph control loop and the single-prompt baseline.

ingest -> summarize -> orchestrate -> [critic x5 in parallel] -> evaluate -> (rejected axes -> critic) -> synthesize
"""

from __future__ import annotations

import json
import time
import uuid
from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from . import agents, prompts
from .agents import ReviewState, RunContext
from .config import AXES, Settings
from .cost import CostTracker
from .events import Listener, Trace
from .gateway import Approver, AuditLog, ToolGateway, deny_all
from .lessons import LessonStore
from .llm import build_llm
from .report import render_markdown

BASELINE_TOKEN_BUDGET = 3000


def build_graph(ctx: RunContext):
    g = StateGraph(ReviewState)
    g.add_node("ingest", partial(agents.ingest, ctx=ctx))
    g.add_node("summarize", partial(agents.summarize, ctx=ctx))
    g.add_node("orchestrate", partial(agents.orchestrate, ctx=ctx))
    g.add_node("critic", partial(agents.critic, ctx=ctx))
    g.add_node("evaluate", partial(agents.evaluate, ctx=ctx))
    g.add_node("synthesize", partial(agents.synthesize, ctx=ctx))
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "summarize")
    g.add_edge("summarize", "orchestrate")
    g.add_conditional_edges("orchestrate", agents.dispatch, ["critic"])
    g.add_edge("critic", "evaluate")
    g.add_conditional_edges("evaluate", agents.route_after_evaluate, ["critic", "synthesize"])
    g.add_edge("synthesize", END)
    return g.compile()


def _lessons(settings: Settings) -> LessonStore | None:
    return LessonStore.from_settings(settings) if settings.use_memory else None


async def run_review(paper: str, settings: Settings, *, seed: int = 0, focus: str = "",
                     approver: Approver = deny_all, run_id: str | None = None,
                     listener: Listener | None = None) -> dict:
    run_id = run_id or f"{Path(paper).stem}-s{seed}-{uuid.uuid4().hex[:6]}"
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker, trace = CostTracker(), Trace(run_id, listener)
    trace.emit("run_start", "system", run_id=run_id, paper=paper, seed=seed, config=config_dict(settings))
    start = time.perf_counter()
    async with ToolGateway(settings, AuditLog(run_dir / "audit.jsonl"), trace, approver) as gateway:
        ctx = RunContext(settings, build_llm(settings, tracker), gateway, tracker, trace, seed, _lessons(settings))
        state = await build_graph(ctx).ainvoke({"paper_path": paper, "user_focus": focus}, {"recursion_limit": 50})
    final = state["final"]
    final.update({"run_id": run_id, "mode": "full", "seed": seed, "wall_latency_s": round(time.perf_counter() - start, 2),
                  "cost": tracker.summary(), "config": config_dict(settings), "tool_flags": gateway.flags})
    trace.emit("run_end", "system", latency_s=final["wall_latency_s"], cost=final["cost"], accepted=final["accepted"])
    _save(run_dir, final, trace, tracker)
    return final


async def run_baseline(paper: str, settings: Settings, *, seed: int = 0, run_id: str | None = None) -> dict:
    """Simpler configuration: one LLM call over the paper's first sentences; no tools, planning or reflection."""
    run_id = run_id or f"baseline-{Path(paper).stem}-s{seed}-{uuid.uuid4().hex[:6]}"
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker, trace = CostTracker(), Trace(run_id)
    trace.emit("run_start", "system", run_id=run_id, paper=paper, seed=seed, mode="baseline")
    start = time.perf_counter()
    async with ToolGateway(settings, AuditLog(run_dir / "audit.jsonl"), trace) as gateway:
        meta = await gateway.call("ingest", "pdf_ingest", {"path": paper})
        hits = await gateway.call("critic", "search_paper", {"paper_id": meta["paper_id"],
                                                             "query": " ".join(prompts.AXIS_GUIDE.values()), "k": 8})
        lines, used = [], 0
        for h in hits:
            for s in h["sentences"]:
                used += len(s["text"]) // 4
                if used < BASELINE_TOKEN_BUDGET:
                    lines.append(f"[{s['id']}] {s['text']}")
        ctx = RunContext(settings, build_llm(settings, tracker), gateway, tracker, trace, seed)
        raw = await ctx.ask("baseline", "baseline", prompts.baseline_user("\n".join(lines), settings.guardrail),
                            max_tokens=1200)
        known = {line[1:line.index("]")] for line in lines}
        critiques = []
        for c in raw.get("critiques", []) if isinstance(raw.get("critiques"), list) else []:
            if isinstance(c, dict) and c.get("axis") in AXES and c["axis"] not in {x["axis"] for x in critiques}:
                ids = agents.clean_ids(c.get("evidence_ids"))
                found = (await gateway.call("evaluator", "get_sentences", {"paper_id": meta["paper_id"], "ids": ids})
                         )["found"] if ids else {}
                critiques.append({"axis": c["axis"], "weakness": str(c.get("weakness", ""))[:600], "evidence_ids": ids,
                                  "quotes": [{"id": i, **found[i]} for i in ids if i in found],
                                  "question": str(c.get("question", ""))[:300],
                                  "grounded": bool(ids) and all(i in known for i in ids)})
    final = {"paper_id": meta["paper_id"], "title": meta["title"], "critiques": critiques, "run_id": run_id,
             "mode": "baseline", "seed": seed, "wall_latency_s": round(time.perf_counter() - start, 2),
             "cost": tracker.summary(), "config": config_dict(settings)}
    trace.emit("run_end", "system", latency_s=final["wall_latency_s"], cost=final["cost"])
    _save(run_dir, final, trace, tracker)
    return final


def _save(run_dir: Path, final: dict, trace: Trace, tracker: CostTracker) -> None:
    (run_dir / "output.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(final), encoding="utf-8")
    trace.save(run_dir / "trace.jsonl")
    tracker.dump(run_dir / "cost.json")


def config_dict(s: Settings) -> dict:
    return {"provider": s.llm_provider, "model": s.llm_model, "temperature": s.temperature,
            "max_reflections": s.max_reflections, "accept_score": s.accept_score, "guardrail": s.guardrail,
            "retrieval": s.retrieval, "use_memory": s.use_memory, "use_graph": s.use_graph}
