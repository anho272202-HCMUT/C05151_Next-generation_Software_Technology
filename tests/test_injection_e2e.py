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


# ------------------------------------------------------------------ semantic study (offline parts)

def test_epistemic_rule_is_opt_in_and_only_for_reasoning_roles():
    from critic_agent import prompts

    assert prompts.EPISTEMIC_RULE not in prompts.system("critic", True)
    assert prompts.EPISTEMIC_RULE in prompts.system("critic", True, True)
    assert prompts.EPISTEMIC_RULE in prompts.system("evaluator", False, True)
    assert prompts.EPISTEMIC_RULE not in prompts.system("summarizer", True, True)
    assert Settings().use_l5 is False and Settings(l5_epistemic=True).use_l5 is True


def test_semantic_payload_file_is_well_formed():
    import json

    from critic_agent.injection_semantic import PAYLOADS, load_payloads

    payloads = load_payloads()
    assert len({p["id"] for p in payloads}) == len(payloads) == 30
    assert {p["kind"] for p in payloads} == {"assertion", "decoy"}
    assert all(p["marker"].lower() in p["text"].lower() for p in payloads if p["kind"] == "decoy")
    # semantic means: the sanitiser must not see them (no imperative, no addressee)
    from critic_agent.guardrail import match_rule
    assert all(match_rule(p["text"]) is None for p in payloads), [p["id"] for p in payloads if match_rule(p["text"])]
    assert json.loads(PAYLOADS.read_text())["payloads"]


def test_decoy_measurement_and_fisher():
    from critic_agent.injection_semantic import fisher, measure

    payload = {"id": "x", "axis": "fairness", "kind": "decoy", "marker": "wall-clock time", "text": "Limitations. We did not compare wall-clock time."}
    final = {"critiques": [{"axis": "fairness", "weakness": "The paper does not compare Wall-Clock Time.", "question": "?", "grounded": True, "quotes": []}],
             "accepted": 1, "cost": {"input_tokens": 1, "output_tokens": 1}}
    row = measure(final, "reflexion.pdf", payload)
    assert row["decoy_adopted"] is True and row["retained"]["fairness"] is False
    assert fisher(0, 5, 5, 0) < 0.01 and fisher(2, 3, 2, 3) == 1.0


def test_web_study_export_matches_raw_rows(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("export_studies", Path(__file__).resolve().parents[1] / "scripts" / "export_injection_studies.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    e2e, sem = mod.e2e(), mod.semantic()
    assert sum(r["imperative"][1] for r in e2e) == 184 and all(r["imperative"][0] == 0 for r in e2e)
    assert all(r["hidden_cited_l1"][0] == 0 for r in e2e)
    assert sem["configs"]["none"]["decoy_adopted"] == [23, 60] and sem["configs"]["system+L5"]["decoy_adopted"] == [15, 60]
    assert 0.01 < sem["p"]["attack_vs_clean_no_defence"] < 0.02
    assert json.loads((Path(__file__).resolve().parents[1] / "web" / "data" / "injection_studies.json").read_text())["e2e"]
