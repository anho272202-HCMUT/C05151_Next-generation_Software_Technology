import pytest

from critic_agent.config import load_settings


@pytest.fixture
def mock_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    return load_settings().with_overrides(llm_provider="mock", runs_dir=tmp_path / "runs",
                                          cache_dir=tmp_path / "cache", memory_dir=tmp_path / "memory",
                                          graph_path=tmp_path / "graph.json")
