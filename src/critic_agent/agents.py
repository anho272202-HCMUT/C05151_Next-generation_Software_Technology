"""Agent nodes: ingest, Summarizer, Orchestrator, per-axis Critic workers, Evaluator, Meta-reviewer, Synthesizer."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.types import Send

from . import prompts
from .config import AXES, Settings
from .cost import CostTracker
from .events import Trace
from .gateway import ToolDenied, ToolError, ToolGateway
from .guardrail import sanitize
from .llm import LLMClient, parse_json
from .memory import EvidenceWindow

MAX_LLM_ATTEMPTS = 3
MIN_WEAKNESS_WORDS = 8
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
RESPONSE_SEARCH_K = 4  # chunks searched per weakness for passages where the authors discuss it
RESPONSE_MAX_SENTENCES = 12
GRAPH_AXES = {"fairness", "generalizability", "data_reliability", "reproducibility"}


class ReviewState(TypedDict, total=False):
    paper_path: str
    user_focus: str
    paper_id: str
    title: str
    num_pages: int
    first_pages: str
    code_urls: list[str]
    flags: list[dict]
    card: dict
    plan: dict[str, dict]
    graph_facts: str
    lessons: list[str]
    round: int
    pending: list[str]
    critiques: Annotated[list[dict], operator.add]  # every attempt from every worker
    best: dict[str, dict]
    feedback: dict[str, str]
    assessment: dict
    final: dict


@dataclass
class RunContext:
    settings: Settings
    llm: LLMClient
    gateway: ToolGateway
    tracker: CostTracker
    trace: Trace
    seed: int
    lessons_store: Any = None  # lessons.LessonStore, optional
    _calls: int = 0

    async def ask(self, role: str, agent: str, user: str, max_tokens: int = 700,
                  temperature: float | None = None) -> dict:
        """LLM call with JSON parsing; retries with a fresh seed on backend errors or malformed JSON.

        Returns {} after MAX_LLM_ATTEMPTS so one bad generation degrades a step instead of crashing the run.
        """
        system = prompts.system(role, self.settings.use_l3, self.settings.use_l5)
        for attempt in range(MAX_LLM_ATTEMPTS):
            self._calls += 1
            try:
                resp = await self.llm.complete(system, user, agent=agent, seed=self.seed * 1000 + self._calls,
                                               max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # timeouts, 5xx, degenerate generation
                self.trace.emit("llm_error", agent, attempt=attempt, error=f"{type(exc).__name__}: {exc}"[:200])
                continue
            self.trace.emit("llm_call", agent, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                            latency_s=round(resp.latency_s, 2))
            try:
                return parse_json(resp.text)
            except (ValueError, json.JSONDecodeError) as exc:
                self.trace.emit("llm_error", agent, attempt=attempt, error=f"bad JSON: {exc}"[:200])
        return {}


# ------------------------------------------------------------------ nodes

async def ingest(state: ReviewState, ctx: RunContext) -> ReviewState:
    ctx.trace.emit("node_start", "ingest")
    meta = await ctx.gateway.call("ingest", "pdf_ingest", {"path": state["paper_path"]})
    flags = list(meta["guardrail_flags"])
    focus = state.get("user_focus", "")
    if focus and ctx.settings.use_l2:
        focus, user_flags = sanitize(focus, "user")
        flags += [f.to_dict() for f in user_flags]
    for f in flags:
        ctx.trace.emit("guardrail_flag", "ingest", **f)
    ctx.trace.emit("node_end", "ingest", paper_id=meta["paper_id"], title=meta["title"], pages=meta["num_pages"],
                   sentences=meta["num_sentences"], chunks=meta["num_chunks"], flags=len(flags))
    return {"paper_id": meta["paper_id"], "title": meta["title"], "num_pages": meta["num_pages"],
            "first_pages": meta["first_pages"], "flags": flags, "user_focus": focus,
            "code_urls": [meta["own_repo"]] if meta.get("own_repo") else []}


async def summarize(state: ReviewState, ctx: RunContext) -> ReviewState:
    ctx.trace.emit("node_start", "summarizer")
    hits = await ctx.gateway.call("summarizer", "search_paper", {
        "paper_id": state["paper_id"], "query": "experimental setup baselines benchmark model results", "k": 3})
    setup = "\n".join(s["text"] for h in hits for s in h["sentences"])
    card = await ctx.ask("summarizer", "summarizer",
                         prompts.summarizer_user(state["first_pages"][:ctx.settings.first_pages_chars], setup[:2500],
                                                 ctx.settings.use_l3))
    if state.get("title") and state["title"] != "Untitled":  # metadata comes from the PDF, never from the LLM
        card["title"] = state["title"]
    card["domain"] = card.get("domain") if card.get("domain") in prompts.DOMAINS else "general"
    ctx.trace.emit("node_end", "summarizer", card=card)
    return {"card": card}


async def orchestrate(state: ReviewState, ctx: RunContext) -> ReviewState:
    ctx.trace.emit("node_start", "orchestrator")
    lessons = []
    if ctx.lessons_store is not None and ctx.settings.use_memory:
        lessons = ctx.lessons_store.retrieve(json.dumps(state["card"])[:800], state["paper_id"])
    raw = await ctx.ask("orchestrator", "orchestrator",
                        prompts.orchestrator_user(state["card"], state.get("user_focus", ""), lessons,
                                                  ctx.settings.use_l3, _domain(state["card"])))
    plan = _enrich_plan(_normalise_plan(raw.get("plan"), _domain(state["card"])), state["card"])
    graph_facts = ""
    if ctx.settings.use_graph and raw.get("graph_lookup", True):
        graph_facts = await _graph_facts(ctx, Path(state["paper_path"]).stem)
        if not graph_facts:  # not in the corpus graph under its file name: try the title
            graph_facts = await _graph_facts(ctx, state["card"].get("title") or state.get("title", ""))
    ctx.trace.emit("node_end", "orchestrator", plan=plan, graph_lookup=bool(graph_facts), lessons=len(lessons))
    return {"plan": plan, "graph_facts": graph_facts, "lessons": lessons, "round": 0, "pending": list(AXES),
            "best": {}, "feedback": {}}


def dispatch(state: ReviewState) -> list[Send]:
    """Fan out one critic worker per pending axis (all axes in round 0, only rejected ones afterwards)."""
    out = []
    for axis in state["pending"]:
        out.append(Send("critic", {
            "axis": axis, "round": state["round"], "paper_id": state["paper_id"], "card": state["card"],
            "queries": state["plan"][axis]["queries"], "graph_facts": state.get("graph_facts", ""),
            "lessons": state.get("lessons", []), "feedback": state.get("feedback", {}).get(axis, ""),
            "previous": state.get("best", {}).get(axis), "code_urls": state.get("code_urls", [])}))
    return out


async def critic(task: dict, ctx: RunContext) -> ReviewState:
    axis, rnd = task["axis"], task["round"]
    agent = f"critic:{axis}"
    ctx.trace.emit("node_start", agent, round=rnd)
    window = EvidenceWindow(ctx.settings.evidence_token_budget)
    queries = list(task["queries"])
    if task.get("previous"):  # on a retry, also search around what the evaluator rejected
        queries.append(task["previous"].get("weakness", "")[:200])
    for query in [q for q in queries if q.strip()]:
        try:
            hits = await ctx.gateway.call("critic", "search_paper",
                                          {"paper_id": task["paper_id"], "query": query, "k": ctx.settings.search_k})
        except (ToolDenied, ToolError) as exc:
            ctx.trace.emit("tool_error", agent, error=str(exc)[:200])
            continue
        for chunk in hits:
            for s in chunk["sentences"]:
                window.add(s["id"], s["text"])
    graph = task["graph_facts"] if axis in GRAPH_AXES else ""
    if axis == "reproducibility":
        graph = "\n".join(x for x in (graph, await _artifact_facts(ctx, agent, task.get("code_urls", []))) if x)
    raw = await ctx.ask("critic", agent, prompts.critic_user(
        axis, task["card"], window.render(), graph, task["lessons"], task["feedback"], task.get("previous"),
        ctx.settings.use_l3, _domain(task["card"])))
    crit = _clean_critique(raw, axis, rnd, window)
    ctx.trace.emit("critique", agent, **{k: v for k, v in crit.items() if k != "window_ids"})
    ctx.trace.emit("node_end", agent, round=rnd, evidence=len(window), evidence_tokens=window.tokens)
    return {"critiques": [crit]}


async def evaluate(state: ReviewState, ctx: RunContext) -> ReviewState:
    rnd = state["round"]
    ctx.trace.emit("node_start", "evaluator", round=rnd)
    current = [c for c in state["critiques"] if c["round"] == rnd]
    guides = prompts.axis_guides(_domain(state.get("card", {})))
    to_judge = []
    for c in current:
        c["quotes"] = await _quotes(ctx, state["paper_id"], c["evidence_ids"])
        c["grounded"], c["ground_issue"] = _grounding(c)
        if c["grounded"]:
            to_judge.append({"id": c["axis"], "axis": c["axis"], "axis_definition": guides[c["axis"]],
                             "weakness": c["weakness"],
                             "cited_sentences": [q["text"] for q in c["quotes"]], "question": c["question"]})
    judged = {}
    if to_judge:
        raw = await ctx.ask("evaluator", "evaluator", prompts.evaluator_user(to_judge), max_tokens=600,
                            temperature=ctx.settings.judge_temperature)
        judged = {s.get("id"): s for s in raw.get("scores", []) if isinstance(s, dict)}
    best, feedback, retry = dict(state.get("best", {})), {}, []
    for c in current:
        verdict = judged.get(c["axis"], {})
        c["score"] = _as_float(verdict.get("score"), 0.0) if c["grounded"] else 0.0
        c["supported"] = bool(verdict.get("supported", False)) if c["grounded"] else False
        c["accepted"] = c["grounded"] and c["supported"] and c["score"] >= ctx.settings.accept_score
        c["feedback"] = c["ground_issue"] or verdict.get("feedback") or ("" if c["accepted"] else "Be more specific.")
        prev = best.get(c["axis"])
        improved = prev is None or _rank(c) > _rank(prev)
        if improved:
            best[c["axis"]] = c
        ctx.trace.emit("evaluation", "evaluator", axis=c["axis"], round=rnd, score=c["score"],
                       grounded=c["grounded"], supported=c["supported"], accepted=c["accepted"],
                       feedback=c["feedback"][:200])
        # Targeted reflection: retry only rejected axes, only while attempts help, within budget.
        if not c["accepted"] and not c.get("no_issue") and rnd < ctx.settings.max_reflections and improved:
            retry.append(c["axis"])
            feedback[c["axis"]] = c["feedback"]
    if retry:
        ctx.trace.emit("reflection", "evaluator", round=rnd + 1, axes=retry)
    ctx.trace.emit("node_end", "evaluator", round=rnd, accepted=sum(c["accepted"] for c in current),
                   judged=len(to_judge), retry=retry)
    return {"best": best, "feedback": feedback, "pending": retry, "round": rnd + 1}


def route_after_evaluate(state: ReviewState) -> list[Send] | str:
    return dispatch(state) if state.get("pending") else "meta_review"


async def meta_review(state: ReviewState, ctx: RunContext) -> ReviewState:
    """Overall 1-10 assessment of the paper. Separate from the critique scores, which grade the critiques.

    Before scoring, an author-response step reads, for every accepted weakness, the passages of the paper that
    discuss it: a weakness the paper refutes, already addresses, or openly acknowledges and bounds must not weigh
    like an unanswered flaw (a critic only sees the sentences that support its weakness).
    """
    if not ctx.settings.paper_score:
        return {"assessment": {}}
    ctx.trace.emit("node_start", "meta_reviewer")
    critiques = [state["best"][a] for a in AXES if a in state.get("best", {})]
    await _author_response(state, ctx, [c for c in critiques if c.get("accepted")])
    raw = await ctx.ask("meta_reviewer", "meta_reviewer", prompts.meta_reviewer_user(
        state["card"], state["first_pages"][:ctx.settings.first_pages_chars], critiques, ctx.settings.use_l3),
        max_tokens=1200, temperature=ctx.settings.judge_temperature)
    assessment = _clean_assessment(raw)
    ctx.trace.emit("node_end", "meta_reviewer", overall=assessment.get("overall"),
                   dimensions=assessment.get("dimensions"))
    return {"assessment": assessment}


async def _author_response(state: ReviewState, ctx: RunContext, critiques: list[dict]) -> None:
    """Attach `response` {accurate, acknowledged, addressed, note} to each critique from the paper's own text."""
    items = []
    for c in critiques:
        cited = set(c.get("evidence_ids", []))
        try:
            hits = await ctx.gateway.call("meta_reviewer", "search_paper", {
                "paper_id": state["paper_id"], "query": c["weakness"][:300], "k": RESPONSE_SEARCH_K})
        except (ToolDenied, ToolError) as exc:
            ctx.trace.emit("tool_error", "meta_reviewer", error=str(exc)[:200])
            hits = []
        context = [f"[{x['id']}] {x['text']}" for h in hits for x in h["sentences"] if x["id"] not in cited]
        items.append({"id": c["axis"], "weakness": c["weakness"],
                      "cited_sentences": [q["text"] for q in c.get("quotes", [])],
                      "paper_context": context[:RESPONSE_MAX_SENTENCES]})
    if not items:
        return
    raw = await ctx.ask("evaluator", "meta_reviewer", prompts.author_response_user(items, ctx.settings.use_l3),
                        max_tokens=900, temperature=ctx.settings.judge_temperature)
    verdicts = {v.get("id"): v for v in raw.get("responses", []) if isinstance(v, dict)}
    for c in critiques:
        c["response"] = _clean_response(verdicts.get(c["axis"], {}))
        ctx.trace.emit("author_response", "meta_reviewer", axis=c["axis"], **c["response"])


def _clean_response(raw: dict) -> dict:
    """Unknown or malformed answers keep the weakness as it is (accurate, not acknowledged, not addressed)."""
    addressed = raw.get("addressed") if raw.get("addressed") in ("no", "partly", "fully") else "no"
    return {"accurate": raw.get("accurate") is not False, "acknowledged": raw.get("acknowledged") is True,
            "addressed": addressed, "note": str(raw.get("note", ""))[:300]}


async def synthesize(state: ReviewState, ctx: RunContext) -> ReviewState:
    ctx.trace.emit("node_start", "synthesizer")
    critiques = [state["best"][a] for a in AXES if a in state.get("best", {})]
    ranked = sorted([c for c in critiques if c["accepted"]],
                    key=lambda c: (SEVERITY_RANK.get(c["severity"], 1), c["score"]), reverse=True)
    final = {
        "paper_id": state["paper_id"], "title": state["card"].get("title") or state.get("title"),
        "card": state["card"], "plan": state["plan"], "graph_facts": state.get("graph_facts", ""),
        "critiques": [_public(c) for c in critiques],
        "debate_questions": [c["question"] for c in ranked[:3] if c["question"]],
        "rounds": state["round"], "accepted": sum(c["accepted"] for c in critiques),
        "no_issue": sum(bool(c.get("no_issue")) for c in critiques), "assessment": state.get("assessment", {}),
        "attempts": len(state["critiques"]), "guardrail_flags": state.get("flags", []),
    }
    if ctx.lessons_store is not None and ctx.settings.use_memory:
        final["lessons_written"] = ctx.lessons_store.write_from_run(state["paper_id"], state["critiques"])
    ctx.trace.emit("node_end", "synthesizer", accepted=final["accepted"], rounds=final["rounds"],
                   questions=len(final["debate_questions"]))
    return {"final": final}


# ------------------------------------------------------------------ helpers

def _domain(card: dict) -> str:
    domain = card.get("domain") if isinstance(card, dict) else None
    return domain if domain in prompts.DOMAINS else "general"


def _normalise_plan(raw: Any, domain: str = "coding_agent") -> dict[str, dict]:
    plan = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("axis") in AXES:
            queries = [q for q in item.get("queries", []) if isinstance(q, str) and q.strip()][:3]
            plan[item["axis"]] = {"priority": item.get("priority", "medium"), "queries": queries}
    for axis in AXES:  # fall back to the axis description so no worker starts without queries
        entry = plan.setdefault(axis, {"priority": "medium", "queries": []})
        if not entry["queries"]:
            entry["queries"] = [prompts.axis_guides(domain)[axis].split("(")[1].rstrip(")")[:120]]
    return plan


def _enrich_plan(plan: dict[str, dict], card: dict) -> dict[str, dict]:
    """Add paper-specific queries built from the paper card (baseline / benchmark / model names).

    Small models tend to copy the generic axis wording; anchoring queries on entities of *this* paper
    is what makes retrieval land on the comparison tables and evaluation sections.
    """
    def names(key: str) -> list[str]:
        return [str(x)[:60] for x in card.get(key, []) if isinstance(x, (str, int))][:3]

    benchmarks, baselines, models = names("benchmarks"), names("baselines"), names("models")
    coding = _domain(card) == "coding_agent"
    extra = {
        "assumptions": (["environment setup docker dependencies test suite"] if coding
                        else ["environment setup simulated idealised setting"]) + ["we assume"],
        "fairness": [f"compared with {b} results" for b in baselines] + ["same model budget cost comparison"],
        "data_reliability": [f"{b} dataset contamination leakage training data" for b in benchmarks]
                            + ["evaluation set size filtered subset"],
        "generalizability": [f"{b} programming language repositories" if coding else f"{b} other domains tasks"
                             for b in benchmarks] + ["threats to validity limitations"],
        "reproducibility": ["repeat experiments randomness runs variance", "code released replication package"]
                           + [f"{m} version" for m in models[:1]],
    }
    for axis, queries in extra.items():
        merged = plan[axis]["queries"] + [q for q in queries if q not in plan[axis]["queries"]]
        plan[axis]["queries"] = merged[:5]
    return plan


async def _graph_facts(ctx: RunContext, title: str) -> str:
    try:
        res = await ctx.gateway.call("orchestrator", "graph_query", {"paper": title[:120]})
    except (ToolDenied, ToolError):
        return ""
    lines = []
    profile = (res.get("info") or {}).get("ref_profile")
    if profile:
        lines.append(f"SOURCES OF THIS PAPER: {profile['total']} references, {profile['peer_reviewed']} peer-reviewed, "
                     f"{profile['preprint']} preprints, {profile['web']} web/software links "
                     f"(median year {profile['median_year']})")
    for n in res.get("neighbors", []):
        meta = ", ".join(str(x) for x in (n.get("venue"), n.get("year")) if x)
        tag = "" if n.get("verified", True) else " (unverified proposal)"
        lines.append(f"{n['relation']}{tag} ({n['direction']}): {n['title']}" + (f" [{meta}]" if meta else "") +
                     (f" — {n['note']}" if n.get("note") else ""))
    return "\n".join(lines)


def clean_ids(raw: Any) -> list[str]:
    """Normalise model-written sentence ids ('[p7s3]', ' p7s3 ') and keep at most three."""
    return [str(i).strip("[] ") for i in (raw if isinstance(raw, list) else []) if isinstance(i, (str, int))][:3]


async def _artifact_facts(ctx: RunContext, agent: str, urls: list[str]) -> str:
    if not urls:
        return "ARTIFACT CHECK: no GitHub repository of this paper is linked in the PDF."
    try:
        res = await ctx.gateway.call("critic", "artifact_check", {"url": urls[0]})
    except (ToolDenied, ToolError) as exc:
        ctx.trace.emit("tool_error", agent, error=str(exc)[:200])
        return ""
    if not res.get("exists"):
        return f"ARTIFACT CHECK: {res['url']} does not exist (404)."
    return (f"ARTIFACT CHECK of {res['url']}: environment files {res['environment_files'] or 'none'}; "
            f"README mentions seeds: {res['readme_mentions_seed']}; mentions reproduction: "
            f"{res['readme_mentions_reproduce']}; licence {res['license']}; last push {res['last_push']}; "
            f"archived {res['archived']}.")


def _clean_critique(raw: dict, axis: str, rnd: int, window: EvidenceWindow) -> dict:
    ids = clean_ids(raw.get("evidence_ids"))
    severity = raw.get("severity") if raw.get("severity") in SEVERITY_RANK else "medium"
    return {"axis": axis, "round": rnd, "no_issue": bool(raw.get("no_issue")) or not raw,
            "weakness": str(raw.get("weakness") or raw.get("reason") or "")[:600], "evidence_ids": ids,
            "question": str(raw.get("question", ""))[:300], "severity": severity,
            "window_ids": sorted(window.ids), "fabricated_ids": [i for i in ids if i not in window.ids]}


async def _quotes(ctx: RunContext, paper_id: str, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    res = await ctx.gateway.call("evaluator", "get_sentences", {"paper_id": paper_id, "ids": ids})
    return [{"id": i, **res["found"][i]} for i in ids if i in res["found"]]


def _grounding(c: dict) -> tuple[bool, str]:
    """Deterministic checks before any LLM judgement."""
    if c["no_issue"]:
        return False, "Critic reported no issue on this axis."
    if not c["evidence_ids"]:
        return False, "No evidence sentence cited; cite ids from EVIDENCE."
    if c["fabricated_ids"]:
        return False, f"Cited ids not in your evidence: {', '.join(c['fabricated_ids'])}."
    if len(c["quotes"]) != len(c["evidence_ids"]):
        return False, "Some cited ids do not exist in the paper."
    if len(c["weakness"].split()) < MIN_WEAKNESS_WORDS:
        return False, "Weakness is too short to be specific."
    return True, ""


def _rank(c: dict) -> tuple:
    return (c.get("accepted", False), c.get("grounded", False), c.get("score", 0.0))


def _public(c: dict) -> dict:
    keep = ("axis", "round", "weakness", "evidence_ids", "quotes", "question", "severity", "score", "grounded",
            "supported", "accepted", "feedback", "no_issue", "response")
    return {k: c.get(k) for k in keep}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_assessment(raw: dict) -> dict:
    """Keep only well-formed scores in 1-10; an empty or malformed answer gives an empty assessment."""
    def score(value: Any) -> float | None:
        v = _as_float(value, -1.0)
        return round(v, 1) if 1.0 <= v <= 10.0 else None

    dims = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    dimensions = {d: s for d in prompts.META_DIMENSIONS if (s := score(dims.get(d))) is not None}
    overall = score(raw.get("overall"))
    if overall is None:
        return {}
    strengths = [str(x)[:200] for x in raw.get("strengths", []) if isinstance(x, str)][:4] \
        if isinstance(raw.get("strengths"), list) else []
    confidence = _as_float(raw.get("confidence"), 0.0)
    return {"overall": overall, "dimensions": dimensions, "strengths": strengths,
            "confidence": confidence if 1 <= confidence <= 5 else None,
            "justification": str(raw.get("justification", ""))[:800]}
