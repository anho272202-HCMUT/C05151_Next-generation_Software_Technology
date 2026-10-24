from critic_agent.agents import _clean_critique, _enrich_plan, _grounding, _normalise_plan
from critic_agent.config import AXES
from critic_agent.memory import EvidenceWindow


def _window():
    w = EvidenceWindow(token_budget=200)
    w.add("p7s1", "We compare with Devin using its technical report.")
    w.add("p7s2", "All runs use GPT-4 with temperature zero.")
    return w


def test_evidence_window_respects_budget():
    w = EvidenceWindow(token_budget=20)
    assert w.add("a", "x" * 40) and not w.add("b", "y" * 80)
    assert w.ids == {"a"} and w.dropped == 1


def test_fabricated_ids_fail_grounding():
    c = _clean_critique({"weakness": "Devin numbers are copied from a report, not re-run under the same budget.",
                         "evidence_ids": ["p7s1", "p9s9"], "question": "Same budget?"}, "fairness", 0, _window())
    c["quotes"] = [{"id": "p7s1", "page": 7, "text": "..."}]
    ok, issue = _grounding(c)
    assert not ok and "p9s9" in issue


def test_valid_critique_is_grounded():
    c = _clean_critique({"weakness": "Devin numbers are copied from a report, not re-run under the same budget.",
                         "evidence_ids": ["p7s1"], "question": "Same budget?"}, "fairness", 0, _window())
    c["quotes"] = [{"id": "p7s1", "page": 7, "text": "..."}]
    assert _grounding(c) == (True, "")


def test_empty_llm_output_is_no_issue():
    c = _clean_critique({}, "fairness", 0, _window())
    assert c["no_issue"] and not _grounding(c)[0]


def test_plan_covers_all_axes_and_uses_card_entities():
    plan = _enrich_plan(_normalise_plan([{"axis": "fairness", "queries": ["baseline"]}]),
                        {"baselines": ["Devin"], "benchmarks": ["SWE-bench Lite"], "models": ["GPT-4"]})
    assert set(plan) == set(AXES)
    assert any("Devin" in q for q in plan["fairness"]["queries"])
    assert all(1 <= len(p["queries"]) <= 5 for p in plan.values())
