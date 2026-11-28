"""Runtime configuration from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AXES = ("assumptions", "fairness", "data_reliability", "generalizability", "reproducibility")

HOSTED_PROVIDERS = ("anthropic", "openai")
# Hosted models have a 128k-1M token context, so the small-model budgets (sized for qwen2.5:3b with num_ctx=8192)
# would hide most of a 10-40k-word paper from the critics. Applied only when the matching env var is not set.
HOSTED_BUDGETS = {"evidence_token_budget": 8000, "search_k": 8, "first_pages_chars": 16000}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"  # ollama | anthropic | openai | mock
    llm_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    temperature: float = 0.7
    judge_temperature: float = 0.0  # Evaluator and Meta-reviewer: grading should not depend on sampling noise
    effort: str = "high"  # hosted reasoning models (Anthropic output_config.effort): low | medium | high | xhigh | max
    hosted_min_max_tokens: int = 16000  # thinking counts against max_tokens on hosted reasoning models
    num_ctx: int = 8192
    # Ollama thinking models (qwen3, gpt-oss, deepseek-r1): "" = do not send `think`; "false"/"true"; gpt-oss takes
    # "low" | "medium" | "high". Thinking tokens count against num_predict, so they get their own budget on top.
    ollama_think: str = ""
    ollama_think_tokens: int = 4096
    ollama_json_format: bool = True  # Ollama `format: "json"`; turn off for a model whose output breaks under it
    papers_dir: Path = PROJECT_ROOT / "data" / "papers"
    runs_dir: Path = PROJECT_ROOT / "runs"
    cache_dir: Path = PROJECT_ROOT / "runs" / "cache"
    memory_dir: Path = PROJECT_ROOT / "runs" / "memory"
    graph_path: Path = PROJECT_ROOT / "runs" / "graph" / "paper_graph.json"
    # Behaviour / ablation switches
    guardrail: bool = True  # master switch; the per-layer flags below override it for ablations
    l1_parser: bool | None = None  # L1 hidden-text removal (None = follow `guardrail`)
    l2_sanitizer: bool | None = None  # L2 normalise + redact (PDF text, user request, tool output)
    l3_spotlight: bool | None = None  # L3 <untrusted> tags + security rules in the prompts
    l5_epistemic: bool = False  # L5 epistemic rule: the authors' assertions about their own evaluation are claims, not evidence
    retrieval: str = "hybrid"  # hybrid | bm25
    max_reflections: int = 2  # extra rounds for rejected critiques only
    accept_score: float = 7.0  # evaluator score needed to accept a critique
    use_memory: bool = True
    use_graph: bool = True
    allow_code_execution: bool = False  # execute_code_in_sandbox is off unless explicitly enabled
    search_k: int = 4
    evidence_token_budget: int = 1800  # per critic worker (short-term memory window)
    first_pages_chars: int = 4000  # opening text given to the Summarizer and the Meta-reviewer
    paper_score: bool = True  # Meta-reviewer: overall 1-10 assessment of the paper (separate from critique scores)

    @property
    def hosted(self) -> bool:
        return self.llm_provider in HOSTED_PROVIDERS

    @property
    def use_l1(self) -> bool:
        return self.guardrail if self.l1_parser is None else self.l1_parser

    @property
    def use_l2(self) -> bool:
        return self.guardrail if self.l2_sanitizer is None else self.l2_sanitizer

    @property
    def use_l3(self) -> bool:
        return self.guardrail if self.l3_spotlight is None else self.l3_spotlight

    @property
    def use_l5(self) -> bool:
        return self.l5_epistemic

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _opt_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    return None if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str) -> Path | None:
    raw = os.getenv(name)
    return Path(raw) if raw else None


def _int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def _float(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw else None


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    base = Settings()
    settings = base.with_overrides(
        llm_provider=os.getenv("LLM_PROVIDER"),
        llm_model=os.getenv("LLM_MODEL"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        embed_model=os.getenv("EMBED_MODEL"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        papers_dir=_path("PAPERS_DIR"),
        runs_dir=_path("RUNS_DIR"),
        cache_dir=_path("CACHE_DIR"),
        graph_path=_path("GRAPH_PATH"),
        guardrail=_bool("CRITIC_GUARDRAIL", base.guardrail),
        l5_epistemic=_bool("CRITIC_L5", base.l5_epistemic),
        l1_parser=_opt_bool("CRITIC_L1"), l2_sanitizer=_opt_bool("CRITIC_L2"), l3_spotlight=_opt_bool("CRITIC_L3"),
        retrieval=os.getenv("CRITIC_RETRIEVAL"),
        allow_code_execution=_bool("CRITIC_ALLOW_CODE", base.allow_code_execution),
        effort=os.getenv("LLM_EFFORT"),
        num_ctx=_int("OLLAMA_NUM_CTX"),  # e.g. 16384 for a 14B model on a 16 GB GPU
        ollama_think=os.getenv("OLLAMA_THINK"),
        ollama_think_tokens=_int("OLLAMA_THINK_TOKENS"),
        ollama_json_format=_bool("OLLAMA_JSON_FORMAT", base.ollama_json_format),
        judge_temperature=_float("CRITIC_JUDGE_TEMPERATURE"),
        paper_score=_bool("CRITIC_PAPER_SCORE", base.paper_score),
        evidence_token_budget=_int("CRITIC_EVIDENCE_TOKENS"),
        search_k=_int("CRITIC_SEARCH_K"),
        first_pages_chars=_int("CRITIC_FIRST_PAGES_CHARS"),
    )
    if settings.hosted:  # larger budgets for large-context models, unless set explicitly above
        env = {"evidence_token_budget": "CRITIC_EVIDENCE_TOKENS", "search_k": "CRITIC_SEARCH_K",
               "first_pages_chars": "CRITIC_FIRST_PAGES_CHARS"}
        settings = settings.with_overrides(**{k: v for k, v in HOSTED_BUDGETS.items() if not os.getenv(env[k])})
    return settings
