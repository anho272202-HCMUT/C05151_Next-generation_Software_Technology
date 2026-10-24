import csv
import json
from pathlib import Path

import pytest

import critic_agent.evaluation as ev

PAPERS = Path(__file__).resolve().parents[1] / "data" / "papers"


def test_labels_cover_40_tasks():
    tasks = ev.load_tasks()
    assert len(tasks) == 40
    assert all(ws and all(w["description"] and w["keywords"] for w in ws) for ws in tasks.values())


def test_kappa():
    assert ev._kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert ev._kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_outcomes():
    assert ev._outcome(None, None, "judge") == "no_critique"
    assert ev._outcome({"grounded": False}, None, "judge") == "ungrounded"
    assert ev._outcome({"grounded": True}, {"judge": None}, "judge") == "grounded_no_match"
    assert ev._outcome({"grounded": True}, {"judge": "x"}, "judge") == "solved"


@pytest.mark.skipif(not (PAPERS / "reflexion.pdf").exists(), reason="corpus not downloaded")
async def test_harness_end_to_end_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(ev, "RUNS_FILE", tmp_path / "results" / "eval_runs.jsonl")
    monkeypatch.setattr(ev, "JUDGE_FILE", tmp_path / "results" / "judgments.jsonl")
    monkeypatch.setattr(ev, "HUMAN_FILE", tmp_path / "human.csv")
    await ev.run_all(["full", "baseline"], [1], ["reflexion.pdf"])
    await ev.run_all(["full", "baseline"], [1], ["reflexion.pdf"])  # resumable: nothing re-run
    assert len(ev._jsonl(ev.RUNS_FILE)) == 2
    await ev.judge_all()
    path = ev.write_human_sample(n=4)
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["q1_on_axis_and_supported (yes/no)"] = "yes"
        r["q2_matches_ground_truth (yes/no)"] = "no"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agreement = ev.agreement()
    assert agreement["n_labelled"] == 4
    summary = ev.summarize()
    assert set(summary["configs"]) == {"full", "baseline"}
    assert (ev.RESULTS / "tables" / "main.tex").exists()
