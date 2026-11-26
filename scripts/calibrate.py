"""Calibrate the Meta-reviewer's paper score on a folder of PDFs.

    uv run python scripts/calibrate.py --papers-dir ~/Desktop/Paper --seeds 1 2 3
    uv run python scripts/calibrate.py --papers-dir ~/Desktop/Paper --refs eval/calibration.json

Runs the full review on every PDF, collects the overall 1-10 score per seed and writes
<out-dir>/calibration_<model>.json. Every finished (paper, seed) is appended to calibration_<model>.runs.jsonl at once,
so after a disconnect (e.g. on Colab) the same command skips what is done and continues. With reference scores (e.g. the mean OpenReview rating of each paper) it also reports
the mean absolute error and the Spearman rank correlation. A useful calibration set mixes strong and weak papers:
on strong papers alone, a scorer that gives 8 to everything looks perfect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import traceback
from pathlib import Path

from critic_agent.config import PROJECT_ROOT, load_settings
from critic_agent.pipeline import run_review


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # average ranks for ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return round(cov / (sx * sy), 3) if sx and sy else None


def load_refs(path: Path | None) -> dict[str, float]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["file"].strip(): float(p["reference_score"]) for p in data.get("papers", [])
            if p.get("reference_score") is not None}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--refs", type=Path, default=PROJECT_ROOT / "eval" / "calibration.json")
    ap.add_argument("--target", type=float, default=8.0, help="score a strong paper should reach")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    args = ap.parse_args()

    papers_dir = args.papers_dir.expanduser().resolve()
    settings = load_settings().with_overrides(papers_dir=papers_dir, use_memory=False)
    refs = load_refs(args.refs)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"calibration_{settings.llm_model.replace(':', '-').replace('/', '-')}"
    out_path, runs_path = out_dir / f"{stem}.json", out_dir / f"{stem}.runs.jsonl"
    done = {}
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["file"], r["seed"])] = r
    pdfs = sorted(papers_dir.glob("*.pdf"))
    for n, pdf in enumerate(pdfs, 1):
        name = pdf.name.strip()
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            try:
                final = await run_review(str(pdf), settings, seed=seed)
            except Exception:  # one broken paper must not stop the whole calibration; it is retried on the next run
                print(f"[{n}/{len(pdfs)}] {name[:50]} seed {seed}: FAILED\n{traceback.format_exc()[-1500:]}")
                continue
            a = final.get("assessment") or {}
            rec = {"file": name, "seed": seed, "overall": a.get("overall"), "dimensions": a.get("dimensions"),
                   "domain": final.get("card", {}).get("domain"), "accepted": final["accepted"],
                   "run_id": final["run_id"], "latency_s": final["wall_latency_s"]}
            with runs_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(name, seed)] = rec
            print(f"[{n}/{len(pdfs)}] {name[:50]} seed {seed}: overall {rec['overall']} domain {rec['domain']} "
                  f"accepted {rec['accepted']}/5 ({rec['latency_s']:.0f}s)", flush=True)

    rows = []
    for pdf in pdfs:
        name = pdf.name.strip()
        recs = [done[(name, s)] for s in args.seeds if (name, s) in done]
        scores = [r["overall"] for r in recs if r["overall"] is not None]
        rows.append({"file": name, "scores": scores, "domain": recs[0]["domain"] if recs else None,
                     "mean": round(statistics.mean(scores), 2) if scores else None,
                     "std": round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0,
                     "reference": refs.get(name)})

    scored = [r for r in rows if r["mean"] is not None]
    paired = [r for r in scored if r["reference"] is not None]
    summary = {
        "model": settings.llm_model, "provider": settings.llm_provider, "seeds": args.seeds,
        "papers": len(rows), "scored": len(scored),
        "mean_score": round(statistics.mean(r["mean"] for r in scored), 2) if scored else None,
        f"share_at_or_above_{args.target:g}": round(sum(r["mean"] >= args.target for r in scored) / len(scored), 2)
        if scored else None,
        "mae_vs_reference": round(statistics.mean(abs(r["mean"] - r["reference"]) for r in paired), 2)
        if paired else None,
        "spearman_vs_reference": spearman([r["mean"] for r in paired], [r["reference"] for r in paired]),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"written {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
