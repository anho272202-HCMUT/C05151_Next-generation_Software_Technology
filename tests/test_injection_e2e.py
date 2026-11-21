"""Layer switches and the offline parts of the end-to-end injection study."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from critic_agent.config import Settings
from critic_agent.guardrail import clean_paper
from critic_agent.injection_adaptive import GOALS as ADAPTIVE_GOALS
from critic_agent.injection_adaptive import attacker_prompt, best_payloads
from critic_agent.injection_e2e import ATTACKS, CONFIGS, GOALS, make_variant, one_run, wilson
from critic_agent.pdf_parser import parse_pdf

PAPERS = Path(__file__).resolve().parents[1] / "data" / "papers"
needs_corpus = pytest.mark.skipif(not (PAPERS / "reflexion.pdf").exists(), reason="corpus not downloaded")


def test_layer_flags_follow_the_master_switch_unless_overridden():
    s = Settings()
    assert (s.use_l1, s.use_l2, s.use_l3) == (True, True, True)
    off = s.with_overrides(guardrail=False)
    assert (off.use_l1, off.use_l2, off.use_l3) == (False, False, False)
    mixed = off.with_overrides(l2_sanitizer=True)
    assert (mixed.use_l1, mixed.use_l2, mixed.use_l3) == (False, True, False)
    assert set(CONFIGS) == {"none", "L3", "L2+L3", "L1+L2+L3"}


@needs_corpus
def test_each_layer_removes_what_it_is_responsible_for(tmp_path):
    hidden = parse_pdf(make_variant(PAPERS / "reflexion.pdf", tmp_path / "h.pdf", "claim", "hidden"))
    visible = parse_pdf(make_variant(PAPERS / "reflexion.pdf", tmp_path / "v.pdf", "obey", "visible"))
    marker, token = GOALS["claim"]["marker"], GOALS["obey"]["token"]
    assert marker in " ".join(clean_paper(hidden, hidden=False, sanitizer=False)[0])  # no defence: payload reaches the LLM
    assert marker not in " ".join(clean_paper(hidden, hidden=True, sanitizer=False)[0])  # L1 alone removes hidden text
    assert marker in " ".join(clean_paper(hidden, hidden=False, sanitizer=True)[0])  # L2 alone cannot see a plain claim
    assert token in " ".join(clean_paper(visible, hidden=False, sanitizer=False)[0])
    assert token not in " ".join(clean_paper(visible, hidden=False, sanitizer=True)[0])  # L2 alone redacts the instruction


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 58)
    assert lo == 0.0 and 0.05 < hi < 0.08
    assert wilson(10, 10)[1] == 1.0


def test_adaptive_helpers():
    assert set(ADAPTIVE_GOALS) == {"obey", "suppress", "leak", "inflate"}
    prompt = attacker_prompt("obey", [{"text": "x", "passed": False, "works": None}], 8)
    assert "PAPERCLEARED" in prompt and "[REDACTED] x" in prompt
    result = {"goals": {"g": {"history": [{"text": "long one", "passed": True, "works": True},
                                          {"text": "short", "passed": True, "works": True},
                                          {"text": "bad", "passed": False, "works": True}]}}}
    assert best_payloads(result, 1) == {"g": ["short"]}


@needs_corpus
def test_one_offline_run_records_objective_signals(tmp_path):
    sem = asyncio.Semaphore(1)
    row = asyncio.run(one_run("mock", "L1+L2+L3", "reflexion.pdf", "obey", "visible", 0, tmp_path, tmp_path / "cache", sem))
    assert row["failed"] is False and row["success"] is False and row["flags"] >= 1 and row["accepted"] == 5
    assert len(ATTACKS) == 8
