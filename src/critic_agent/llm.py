"""Provider-agnostic async LLM clients. Every call is cost-logged.

Providers: Ollama (native /api/chat, local), Anthropic, OpenAI, and a deterministic MockLLM so the
whole pipeline and its tests run offline.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .cost import CostTracker


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LLMClient(Protocol):
    model: str

    async def complete(self, system: str, user: str, *, agent: str, seed: int | None = None,
                       max_tokens: int = 900) -> LLMResponse: ...


class OllamaLLM:
    def __init__(self, settings: Settings, tracker: CostTracker):
        self.model = settings.llm_model
        self._url = settings.ollama_base_url.rstrip("/") + "/api/chat"
        self._settings = settings
        self._tracker = tracker
        self._client = httpx.AsyncClient(timeout=900)

    async def complete(self, system, user, *, agent, seed=None, max_tokens=900):
        start = time.perf_counter()
        resp = await self._client.post(self._url, json={
            "model": self.model, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": self._settings.temperature, "seed": seed, "num_ctx": self._settings.num_ctx,
                        "num_predict": max_tokens},
        })
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - start
        in_tok, out_tok = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        self._tracker.record(agent, self.model, in_tok, out_tok, latency)
        return LLMResponse(data["message"]["content"], in_tok, out_tok, latency)


class AnthropicLLM:
    def __init__(self, settings: Settings, tracker: CostTracker):
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.model = settings.llm_model
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._temperature = settings.temperature
        self._tracker = tracker

    async def complete(self, system, user, *, agent, seed=None, max_tokens=900):
        start = time.perf_counter()
        resp = await self._client.messages.create(
            model=self.model, system=system + "\nRespond with one JSON object only.", max_tokens=max_tokens,
            temperature=self._temperature, messages=[{"role": "user", "content": user}])
        latency = time.perf_counter() - start
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        self._tracker.record(agent, self.model, resp.usage.input_tokens, resp.usage.output_tokens, latency)
        return LLMResponse(text, resp.usage.input_tokens, resp.usage.output_tokens, latency)


class OpenAILLM:
    def __init__(self, settings: Settings, tracker: CostTracker):
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = settings.llm_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._temperature = settings.temperature
        self._tracker = tracker

    async def complete(self, system, user, *, agent, seed=None, max_tokens=900):
        start = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self.model, temperature=self._temperature, seed=seed, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        latency = time.perf_counter() - start
        u = resp.usage
        self._tracker.record(agent, self.model, u.prompt_tokens, u.completion_tokens, latency)
        return LLMResponse(resp.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, latency)


class MockLLM:
    """Deterministic offline stand-in that returns schema-valid JSON derived from the prompt."""

    model = "mock"

    def __init__(self, tracker: CostTracker):
        self._tracker = tracker

    async def complete(self, system, user, *, agent, seed=None, max_tokens=900):
        sentence_ids = re.findall(r"\[(p\d+s\d+)\]", user)
        if agent == "summarizer":
            out = {"title": "Mock paper", "task": "program repair", "method": "agent", "benchmarks": ["SWE-bench"],
                   "baselines": ["RAG"], "models": ["GPT-4"], "claims": ["resolves 12% of issues"],
                   "code_url": None}
        elif agent == "orchestrator":
            out = {"plan": [{"axis": a, "priority": "high", "queries": [f"{a} evaluation setup"]}
                            for a in ("assumptions", "fairness", "data_reliability", "generalizability",
                                      "reproducibility")], "graph_lookup": True}
        elif agent.startswith("critic"):
            out = {"weakness": "The comparison may not use the same budget.", "evidence_ids": sentence_ids[:1],
                   "question": "Were all baselines given the same budget?", "severity": "medium"}
        elif agent == "evaluator":
            ids = re.findall(r'"id": "([a-z_]+)"', user)
            out = {"scores": [{"id": i, "score": 8, "supported": True, "feedback": ""} for i in ids]}
        elif agent == "baseline":
            out = {"critiques": [{"axis": "fairness", "weakness": "Unclear budget.", "evidence_ids": sentence_ids[:1],
                                  "question": "Same budget?"}]}
        else:
            out = {}
        text = json.dumps(out)
        tin, tout = approx_tokens(system + user), approx_tokens(text)
        self._tracker.record(agent, self.model, tin, tout, 0.0)
        return LLMResponse(text, tin, tout, 0.0)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_json(text: str) -> dict:
    """Lenient JSON extraction: small local models sometimes wrap JSON in prose or code fences."""
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in model output: {text[:200]!r}") from None
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {"items": value}


def build_llm(settings: Settings, tracker: CostTracker) -> LLMClient:
    providers = {"ollama": OllamaLLM, "anthropic": AnthropicLLM, "openai": OpenAILLM}
    if settings.llm_provider == "mock":
        return MockLLM(tracker)
    if settings.llm_provider not in providers:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
    return providers[settings.llm_provider](settings, tracker)
