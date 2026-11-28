# Critic-Agent — CO5151 Next-generation Software Technology (Group 1)

Multi-agent system that produces **evidence-grounded critical reviews** of research papers on LLM coding agents
(SWE-agent, AutoCodeRover, …) along five axes: assumptions, fairness of comparison, data reliability, generalizability,
reproducibility. Every critique cites a sentence that the system verifies against the PDF.

> Status: D2 (interim) in progress. See `docs/` for the architecture and `report/D2/` for the progress report.

**Live demo (GitHub Pages):** https://anho272202-hcmut.github.io/C05151_Next-generation_Software_Technology/ —
the Critic-Agent Explorer replays real recorded runs step by step (architecture highlighting, tool calls with permission
levels, critiques, evaluator verdicts, targeted reflection), plus the knowledge graph, evaluation results and a
guardrail playground. Run it locally with live reviews: `uv run critic-agent serve` → http://localhost:8000 (Vietnamese guide: `docs/LOCAL_DEMO.md`).

## Repository layout

| Path | Content |
|---|---|
| `src/critic_agent/` | LangGraph pipeline, MCP server + tool gateway, memory, knowledge graph, evaluation |
| `tests/` | Offline tests (mock LLM) |
| `eval/` | Paper corpus manifest, human labels, task definitions, prompt-injection attack set (`injections.json`, `benign.json`) |
| `scripts/` | Download, reproduction and export scripts |
| `web/` | Critic-Agent Explorer (static site for GitHub Pages) |
| `report/D2/` | D2 progress report (LaTeX) |
| `docs/` | Architecture notes, threat model (`threat_model.md`) |
| `results/` | Measured results (`injection_eval*.json`) |

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

## Usage

```bash
uv run python -m critic_agent.graph_builder                       # build the paper graph once (runs/graph/)
uv run critic-agent review swe-agent.pdf --seed 1                 # full multi-agent review -> runs/<run_id>/
uv run critic-agent review autocoderover.pdf --focus "focus on fairness"
uv run critic-agent baseline swe-agent.pdf --seed 1               # simpler config: one LLM call, no tools/loop
uv run pytest -q                                                  # offline tests (mock LLM)
uv run critic-agent serve                                         # web Explorer + live runs on http://localhost:8000
uv run python scripts/export_web_data.py                          # refresh web/data from runs/ and results/
uv run python -m critic_agent.injection_eval --export-web      # measure the prompt-injection defences, refresh web/data/injection_eval.json
uv run python scripts/calibrate.py --papers-dir ~/Desktop/Paper --seeds 1 2 3   # paper scores on a PDF folder vs eval/calibration.json
```

Two different scores appear in a review. The **critique score** (0-10 per axis) is the Evaluator's grade of a critique:
on-axis, specific, supported by the cited sentences. The **paper score** (1-10, ICLR/NeurIPS reviewer scale) is the
Meta-reviewer's assessment of the paper, weighing its strengths against the verified weaknesses. A strong paper can
get low critique scores simply because it has few real weaknesses to find.

Before scoring, the Meta-reviewer checks every accepted weakness against the rest of the paper, as an author rebuttal
would: it searches the passages that discuss the weakness and marks it refuted (the critique misreads the paper),
fully or partly addressed, and acknowledged by the authors. A critic only sees the sentences that support its
weakness, so without this step a limitation the paper openly bounds would weigh like a hidden flaw.

The axis definitions depend on the kind of paper, picked by the Summarizer: `coding_agent`, `agent_benchmark`,
`llm_training`, `llm_evaluation` or `general` (`prompts.DOMAIN_AXIS_GUIDES`).

Each run writes `output.json`, `report.md`, `trace.jsonl` (event log replayed by the web Explorer),
`audit.jsonl` (every tool request incl. denials) and `cost.json` (tokens, latency, USD, gpt-4o-mini-equivalent USD).

## Architecture (D2)

`ingest → Summarizer → Orchestrator → 5 axis Critics (parallel, LangGraph Send) → Evaluator → targeted reflection
(rejected axes only, ≤2 rounds, stop when no improvement) → Meta-reviewer (paper score) → Synthesizer`. Critics cite sentence ids; the system copies the
verbatim sentence, and the Evaluator checks ids deterministically before an LLM judge scores axis-fit and support.
All tools go through `ToolGateway` → MCP server. See `docs/architecture.md`.

## Team

| Member | Student ID | Role |
|---|---|---|
| Hồ Bảo An | 2570164 | Design, implementation, evaluation, report |
