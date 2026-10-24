"""Command line: `critic-agent review <paper.pdf>` and `critic-agent baseline <paper.pdf>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import load_settings
from .pipeline import run_baseline, run_review


async def ask_human(agent: str, tool: str, args: dict) -> bool:
    print(f"\n[APPROVAL NEEDED] agent '{agent}' wants '{tool}':\n{json.dumps(args, indent=2)[:800]}")
    if not sys.stdin.isatty():
        print("Non-interactive session: denied.")
        return False
    return input("Approve? [y/N] ").strip().lower() == "y"


def _print_event(event: dict) -> None:
    t, agent = event["type"], event["agent"]
    if t == "node_start":
        print(f"[{event['t']:7.1f}s] > {agent}" + (f" (round {event['round']})" if "round" in event else ""))
    elif t == "tool_denied":
        print(f"[{event['t']:7.1f}s]   ! {agent} denied {event['tool']} ({event['reason']})")
    elif t == "evaluation":
        mark = "accepted" if event["accepted"] else "rejected"
        print(f"[{event['t']:7.1f}s]   {event['axis']}: {mark} (score {event['score']:.0f})")
    elif t == "guardrail_flag":
        print(f"[{event['t']:7.1f}s]   guardrail: neutralised {event['source']} text")


def main(argv: list[str] | None = None) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="critic-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("review", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("paper", help="PDF inside data/papers/")
        p.add_argument("--seed", type=int, default=0)
    serve = sub.add_parser("serve", help="web Explorer with live runs on http://localhost:8000")
    serve.add_argument("--port", type=int, default=8000)
    review = sub.choices["review"]
    review.add_argument("--focus", default="", help="optional user request, e.g. 'focus on fairness'")
    review.add_argument("--no-reflection", action="store_true")
    review.add_argument("--no-guardrail", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server import serve as run_server

        print(f"Critic-Agent Explorer on http://localhost:{args.port}")
        run_server(port=args.port)
        return
    settings = load_settings()
    if args.command == "baseline":
        final = asyncio.run(run_baseline(args.paper, settings, seed=args.seed))
    else:
        settings = settings.with_overrides(max_reflections=0 if args.no_reflection else None,
                                           guardrail=False if args.no_guardrail else None)
        final = asyncio.run(run_review(args.paper, settings, seed=args.seed, focus=args.focus,
                                       approver=ask_human, listener=_print_event))
    print(open(settings.runs_dir / final["run_id"] / "report.md", encoding="utf-8").read())
    c = final["cost"]
    print(f"run_id={final['run_id']} latency={final['wall_latency_s']}s llm_calls={c['calls']} "
          f"tokens={c['input_tokens']}+{c['output_tokens']} usd={c['usd']} (gpt-4o-mini equiv ${c['usd_equiv']})")
