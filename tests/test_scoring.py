"""Domain axis profiles, the Meta-reviewer paper score and the hosted-model client settings (offline)."""

import asyncio
import types

import pytest

from critic_agent import prompts
from critic_agent.agents import RunContext, _clean_assessment, _enrich_plan, _normalise_plan, meta_review
from critic_agent.config import AXES, HOSTED_BUDGETS, Settings, load_settings
from critic_agent.cost import CostTracker
from critic_agent.events import Trace
from critic_agent.llm import AnthropicLLM, MockLLM
from critic_agent.report import render_markdown


def test_every_domain_defines_all_axes_with_example_queries():
    for domain, guides in prompts.DOMAIN_AXIS_GUIDES.items():
        assert set(guides) == set(AXES), domain
        for axis in AXES:  # _normalise_plan falls back to the text in parentheses
            assert "(" in guides[axis] and guides[axis].rstrip().endswith(")"), (domain, axis)


def test_unknown_domain_falls_back_to_general():
    assert prompts.axis_guides("quantum") is prompts.DOMAIN_AXIS_GUIDES["general"]
    assert prompts.axis_guides(None) is prompts.DOMAIN_AXIS_GUIDES["general"]


def test_critic_prompt_uses_the_domain_guide():
    text = prompts.critic_user("assumptions", {}, "[p1s1] x", "", [], "", None, True, domain="llm_training")
    assert "reward model" in text and "docker" not in text.lower()


def test_plan_for_non_coding_paper_does_not_search_for_docker():
    card = {"domain": "llm_training", "benchmarks": ["GSM8K"], "baselines": ["PPO"], "models": ["DeepSeekMath"]}
    plan = _enrich_plan(_normalise_plan([], "llm_training"), card)
    queries = " ".join(q for p in plan.values() for q in p["queries"]).lower()
    assert "docker" not in queries and "programming language" not in queries
    assert any("PPO" in q for q in plan["fairness"]["queries"])


def test_clean_assessment_keeps_valid_scores_only():
    a = _clean_assessment({"overall": "8", "dimensions": {"soundness": 9, "clarity": 11, "bogus": 5},
                           "strengths": ["a", 3, "b"], "confidence": 4, "justification": "ok"})
    assert a["overall"] == 8.0 and a["dimensions"] == {"soundness": 9.0}
    assert a["strengths"] == ["a", "b"] and a["confidence"] == 4
    assert _clean_assessment({}) == {} and _clean_assessment({"overall": 0}) == {}


def test_meta_prompt_marks_unverified_and_neutral_axes():
    crits = [{"axis": "fairness", "accepted": True, "severity": "high", "weakness": "Baselines differ."},
             {"axis": "assumptions", "accepted": False, "weakness": "Maybe."},
             {"axis": "reproducibility", "no_issue": True}]
    text = prompts.meta_reviewer_user({"title": "T"}, "We propose X.", crits, True)
    assert "VERIFIED, severity high" in text and "UNVERIFIED" in text and "no significant weakness" in text
    assert "8: clear accept" in text and "fame must not change it" in text


def test_meta_review_node_returns_assessment(tmp_path):
    settings = Settings(llm_provider="mock", runs_dir=tmp_path)
    tracker = CostTracker()
    ctx = RunContext(settings, MockLLM(tracker), None, tracker, Trace("t"), seed=1)
    state = {"card": {"title": "T"}, "first_pages": "We propose X.", "best": {}}
    out = asyncio.run(meta_review(state, ctx))
    assert out["assessment"]["overall"] == 7.0
    off = RunContext(settings.with_overrides(paper_score=False), MockLLM(tracker), None, tracker, Trace("t"), seed=1)
    assert asyncio.run(meta_review(state, off)) == {"assessment": {}}


def test_report_shows_paper_score_apart_from_critique_scores():
    final = {"paper_id": "p", "title": "T", "mode": "full", "accepted": 1, "rounds": 1, "attempts": 5,
             "assessment": {"overall": 8.5, "dimensions": {"soundness": 8}, "strengths": ["Big benchmark"],
                            "confidence": 4, "justification": "Strong."},
             "critiques": [{"axis": "fairness", "weakness": "w", "accepted": True, "score": 9},
                           {"axis": "assumptions", "weakness": "", "no_issue": True, "score": 0}]}
    md = render_markdown(final)
    assert "**Overall: 8.5/10**" in md and "critique score 9/10" in md
    assert "no significant weakness found" in md and "critique score 0" not in md


def test_hosted_provider_gets_larger_budgets(monkeypatch):
    monkeypatch.setattr("critic_agent.config.load_dotenv", lambda *a, **k: None)  # ignore a local .env
    for k in ("CRITIC_EVIDENCE_TOKENS", "CRITIC_SEARCH_K", "CRITIC_FIRST_PAGES_CHARS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    s = load_settings()
    assert s.evidence_token_budget == HOSTED_BUDGETS["evidence_token_budget"]
    monkeypatch.setenv("CRITIC_EVIDENCE_TOKENS", "3000")
    assert load_settings().evidence_token_budget == 3000
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("CRITIC_EVIDENCE_TOKENS")
    assert load_settings().evidence_token_budget == Settings().evidence_token_budget


class _FakeMessages:
    def __init__(self, calls, stop_reason="end_turn"):
        self.calls, self.stop_reason = calls, stop_reason

    async def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(
            stop_reason=self.stop_reason, stop_details=types.SimpleNamespace(category="cyber"),
            usage=types.SimpleNamespace(input_tokens=5, output_tokens=7),
            content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text="{}")])


def _anthropic(model, stop_reason="end_turn"):
    llm = AnthropicLLM(Settings(anthropic_api_key="k", llm_model=model), CostTracker())
    calls = []
    plain, beta = _FakeMessages(calls, stop_reason), _FakeMessages(calls, stop_reason)
    llm._client = types.SimpleNamespace(messages=plain, beta=types.SimpleNamespace(messages=beta))
    return llm, calls, plain, beta


def test_anthropic_client_leaves_room_for_thinking_and_sets_effort():
    llm, calls, _, beta = _anthropic("claude-opus-5-5")
    resp = asyncio.run(llm.complete("s", "u", agent="evaluator", max_tokens=600, temperature=0.0))
    kw = calls[0]
    assert resp.text == "{}" and kw["max_tokens"] == 16000 and kw["output_config"] == {"effort": "high"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw


def test_anthropic_client_without_fallback_support_uses_plain_endpoint():
    llm, calls, plain, _ = _anthropic("claude-haiku-5-5")
    asyncio.run(llm.complete("s", "u", agent="critic"))
    assert "fallbacks" not in calls[0] and "betas" not in calls[0]


def test_anthropic_refusal_raises_so_the_step_is_retried():
    llm, _, _, _ = _anthropic("claude-opus-5-5", stop_reason="refusal")
    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(llm.complete("s", "u", agent="critic"))
