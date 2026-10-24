# Critic-Agent architecture (D2)

## Control loop (LangGraph `StateGraph`, `src/critic_agent/pipeline.py`)

| Node | Agent | Does | Tools (via gateway) |
|---|---|---|---|
| `ingest` | — | parse PDF into numbered sentences, sanitise, index; sanitise user request | `pdf_ingest` |
| `summarize` | Summarizer | paper card: task, method, benchmarks, baselines, models, claims, runs/seeds, code URL | `search_paper` |
| `orchestrate` | Orchestrator | per-axis priority + queries (LLM), enriched with card entities; optional graph lookup | `graph_query` |
| `critic` ×5 | Critic workers | one per axis, run in parallel with `Send`; fill an evidence window, write one critique citing sentence ids | `search_paper` |
| `evaluate` | Evaluator | deterministic grounding (ids exist, came from the worker's evidence, weakness long enough) + LLM judge (axis fit, support, specificity); decides which axes to retry | `get_sentences` |
| `synthesize` | Synthesizer | best attempt per axis, top-3 debate questions, Markdown report | — |

Targeted reflection: only rejected axes go back to `critic`, with the evaluator's feedback, at most
`max_reflections` (2) extra rounds, and only while the axis keeps improving. The best attempt per axis is kept.

State fields: `paper_id, card, plan, graph_facts, lessons, round, pending, critiques (append-only reducer), best,
feedback, final`.

## Tool permission levels (`src/critic_agent/permissions.py`)

| Tool | Level | Agents | Control |
|---|---|---|---|
| `pdf_ingest` | read-only | ingest | path allow-list (`data/papers/`), PDF opened read-only; only the server cache is written |
| `search_paper` | read-only | summarizer, orchestrator, critic | k ≤ 8 |
| `get_sentences` | read-only | critic, evaluator, synthesizer | ≤ 12 ids; unknown ids reported as missing |
| `graph_query` | read-only | orchestrator, critic | ≤ 20 nodes, contradictions first |
| `artifact_check` | read-only (open-world) | critic | GET-only GitHub API, results cached; only the paper's own repo |
| `save_annotation` / `delete_annotation` | reversible write | user | note ids allow undo |
| `publish_review` | irreversible write | user | human approval → HMAC token checked by server; append-only, no overwrite |
| `execute_code_in_sandbox` | code execution | critic | disabled by default; human approval + HMAC; Docker `--network none`, 1 GB, 30 s, read-only FS |

Read-only is guaranteed in code: these tools have no write calls, the PDF path is resolved and checked against the
allow-list, and `tests/test_gateway_pipeline.py` hashes the corpus before/after to prove nothing changed.

## Trace events (`events.py`)

`run_start, node_start, node_end, llm_call, llm_error, tool_call, tool_denied, guardrail_flag, critique, evaluation,
reflection, run_end` — saved as `trace.jsonl` and replayed by the web Explorer.

## Knowledge sources and memory (phase 2)

| Source | Built by | Content / scale (8-paper corpus) | Used by |
|---|---|---|---|
| Per-paper index | `pdf_ingest` | 388–1,593 sentences per paper (5,881 total), windows of 4 sentences / stride 3 (1,962 chunks), BM25 + `nomic-embed-text` dense vectors fused by RRF, vectors cached per paper | Summarizer, critics |
| Paper graph | `python -m critic_agent.graph_builder [--propose]` | 17 nodes (8 papers, 9 benchmarks), 36 edges: 17 USES_BENCHMARK, 11 CITES, 6 COMPARES_WITH (with evidence sentences), 2 CONTRADICTS (group-curated, `eval/graph_curated.json`); node venue + reference-source profile | Orchestrator → fairness / data / generalizability / reproducibility critics |
| Reference profile | `references.py` | references split + classified (peer-reviewed / preprint / web / other) — heuristic | graph node attribute, shown to critics |
| Artifact facts | `artifact_check` | repo exists, environment files, README mentions of seeds/reproduction, licence, last push | reproducibility critic |
| Lessons (long-term) | `lessons.py` (ChromaDB) | write only accepted critiques (fix / pattern lessons), sanitised, de-duplicated; retrieve top-3, never from the same paper; `forget(paper_id)` | Orchestrator → critics |

LLM-proposed CONTRADICTS edges (`--propose`) are stored with `verified: false`. With qwen2.5:3b the model proposed
none of the six candidate pairs, including the clear Agentless→SWE-agent case, so the two CONTRADICTS edges in the
graph are group-curated.
