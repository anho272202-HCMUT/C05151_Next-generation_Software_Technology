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
| `graph_query` | read-only | orchestrator, critic | ≤ 20 nodes |
| `save_annotation` / `delete_annotation` | reversible write | user | note ids allow undo |
| `publish_review` | irreversible write | user | human approval → HMAC token checked by server; append-only, no overwrite |
| `execute_code_in_sandbox` | code execution | critic | disabled by default; human approval + HMAC; Docker `--network none`, 1 GB, 30 s, read-only FS |

Read-only is guaranteed in code: these tools have no write calls, the PDF path is resolved and checked against the
allow-list, and `tests/test_gateway_pipeline.py` hashes the corpus before/after to prove nothing changed.

## Trace events (`events.py`)

`run_start, node_start, node_end, llm_call, llm_error, tool_call, tool_denied, guardrail_flag, critique, evaluation,
reflection, run_end` — saved as `trace.jsonl` and replayed by the web Explorer.
