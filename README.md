# Critic-Agent — CO5151 Next-generation Software Technology (Group 1)

Multi-agent system that produces **evidence-grounded critical reviews** of research papers on LLM coding agents
(SWE-agent, AutoCodeRover, …) along five axes: assumptions, fairness of comparison, data reliability, generalizability,
reproducibility. Every critique cites a sentence that the system verifies against the PDF.

> Status: D2 (interim) in progress. See `docs/` for the architecture and `report/D2/` for the progress report.

## Repository layout

| Path | Content |
|---|---|
| `src/critic_agent/` | LangGraph pipeline, MCP server + tool gateway, memory, knowledge graph, evaluation |
| `tests/` | Offline tests (mock LLM) |
| `eval/` | Paper corpus manifest, human labels, task definitions |
| `scripts/` | Download, reproduction and export scripts |
| `web/` | Critic-Agent Explorer (static site for GitHub Pages) |
| `report/D2/` | D2 progress report (LaTeX) |
| `docs/` | Architecture notes and diagrams |

## Setup

Requirements: macOS/Linux, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
cp .env.example .env
uv sync --frozen
./scripts/download_papers.sh
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
```

On a Desktop folder synced by iCloud, keep the virtualenv, papers and run artefacts in `*.nosync` folders
(`.venv -> .venv.nosync`, `data/papers -> data/papers.nosync`, `runs -> runs.nosync`) so iCloud never evicts them mid-run.

## Team

| Member | Student ID | Role |
|---|---|---|
| Hồ Bảo An | 2570164 | Design, implementation, evaluation, report |
