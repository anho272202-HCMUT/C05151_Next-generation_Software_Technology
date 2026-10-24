"""Runtime configuration from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AXES = ("assumptions", "fairness", "data_reliability", "generalizability", "reproducibility")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"  # ollama | anthropic | openai | mock
    llm_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    temperature: float = 0.7
    num_ctx: int = 8192
    papers_dir: Path = PROJECT_ROOT / "data" / "papers"
    runs_dir: Path = PROJECT_ROOT / "runs"
    cache_dir: Path = PROJECT_ROOT / "runs" / "cache"
    memory_dir: Path = PROJECT_ROOT / "runs" / "memory"
    graph_path: Path = PROJECT_ROOT / "runs" / "graph" / "paper_graph.json"
    # Behaviour / ablation switches
    guardrail: bool = True
    retrieval: str = "hybrid"  # hybrid | bm25
    max_reflections: int = 2  # extra rounds for rejected critiques only
    accept_score: float = 7.0  # evaluator score needed to accept a critique
    use_memory: bool = True
    use_graph: bool = True
    allow_code_execution: bool = False  # execute_code_in_sandbox is off unless explicitly enabled
    search_k: int = 4
    evidence_token_budget: int = 1800  # per critic worker (short-term memory window)

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str) -> Path | None:
    raw = os.getenv(name)
    return Path(raw) if raw else None


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    base = Settings()
    return base.with_overrides(
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
        retrieval=os.getenv("CRITIC_RETRIEVAL"),
        allow_code_execution=_bool("CRITIC_ALLOW_CODE", base.allow_code_execution),
    )
