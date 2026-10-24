"""Export recorded runs, the paper graph, evaluation results and system metadata for the web Explorer.

uv run python scripts/export_web_data.py

Picks, per paper, the seed-1 evaluation runs (Full and baseline) when they exist, otherwise the latest CLI run.
Writes web/data/{index,system,graph,summary}.json and web/data/runs/<run_id>.json.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from critic_agent.config import AXES, PROJECT_ROOT, load_settings
from critic_agent.guardrail import INJECTION_PATTERNS
from critic_agent.permissions import TOOL_POLICY
from critic_agent.prompts import AXIS_GUIDE

WEB_DATA = PROJECT_ROOT / "web" / "data"
MAX_TEXT = 600


def _load_run(run_dir: Path) -> dict | None:
    out, trace = run_dir / "output.json", run_dir / "trace.jsonl"
    if not (out.exists() and trace.exists()):
        return None
    final = json.loads(out.read_text())
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    return {"run_id": run_dir.name, "final": final, "events": events,
            "report_md": (run_dir / "report.md").read_text() if (run_dir / "report.md").exists() else ""}


def _candidates(runs_dir: Path) -> dict[tuple[str, str], Path]:
    """(paper stem, mode) -> best run directory."""
    picks: dict[tuple[str, str], Path] = {}
    eval_dir = runs_dir / "eval"
    for d in sorted(eval_dir.glob("eval-*-s1-*")) if eval_dir.exists() else []:
        m = re.match(r"eval-(full|baseline)-s1-(.+)", d.name)
        if m and (d / "output.json").exists():
            picks[(m.group(2), m.group(1))] = d
    for d in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):  # newest first
        if not (d / "output.json").exists() or d.name.startswith(("eval", "smoke")):
            continue
        mode = "baseline" if d.name.startswith("baseline-") else "full"
        stem = re.sub(r"^baseline-", "", re.sub(r"-s\d+-[0-9a-f]{6}$", "", d.name))
        picks.setdefault((stem, mode), d)
    return picks


def _trim(value):
    if isinstance(value, str):
        return value[:MAX_TEXT]
    if isinstance(value, list):
        return [_trim(v) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v) for k, v in value.items()}
    return value


def main() -> None:
    settings = load_settings()
    runs_out = WEB_DATA / "runs"
    if runs_out.exists():
        shutil.rmtree(runs_out)
    runs_out.mkdir(parents=True)
    index = []
    for (stem, mode), run_dir in sorted(_candidates(settings.runs_dir).items()):
        run = _load_run(run_dir)
        if run is None:
            continue
        f = run["final"]
        (runs_out / f"{run['run_id']}.json").write_text(json.dumps(_trim(run), ensure_ascii=False))
        index.append({"run_id": run["run_id"], "paper": stem, "mode": mode, "title": f.get("title"),
                      "seed": f.get("seed"), "accepted": f.get("accepted"), "rounds": f.get("rounds"),
                      "latency_s": f.get("wall_latency_s"), "llm_calls": f["cost"]["calls"],
                      "tokens": f["cost"]["input_tokens"] + f["cost"]["output_tokens"],
                      "events": len(run["events"])})
    (WEB_DATA / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))

    graph = settings.graph_path
    if graph.exists():
        data = json.loads(graph.read_text())
        for n in data["nodes"]:
            n.pop("abstract", None)
        (WEB_DATA / "graph.json").write_text(json.dumps(data, ensure_ascii=False))
    for name in ("summary.json", "judge_agreement.json"):
        src = PROJECT_ROOT / "results" / name
        if src.exists():
            shutil.copy(src, WEB_DATA / name)

    system = {
        "axes": [{"id": a, "guide": AXIS_GUIDE[a]} for a in AXES],
        "tools": [{"name": n, "level": p.level.value, "agents": sorted(p.agents), "needs_approval": p.needs_approval,
                   "description": p.description} for n, p in TOOL_POLICY.items()],
        "injection_patterns": INJECTION_PATTERNS,
        "model": settings.llm_model,
    }
    (WEB_DATA / "system.json").write_text(json.dumps(system, indent=1, ensure_ascii=False))
    print(f"exported {len(index)} runs -> {WEB_DATA}")


if __name__ == "__main__":
    main()
