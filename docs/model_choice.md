# Model choice for D2 (measured 2026-10-24)

Hardware: Apple M1, 8 GB unified memory. Benchmark: `uv run python scripts/bench_models.py qwen2.5:3b qwen2.5:7b`
(one paper-sized prompt, 1,521 input tokens, JSON output, `num_ctx=8192`, seed 1).

| Model (Ollama, Q4) | Prompt speed | Generation speed | Wall time / call |
|---|---|---|---|
| qwen2.5:3b | 265.6 tok/s | 25.7 tok/s | 24.0 s |
| qwen2.5:7b | 39.1 tok/s | 3.6 tok/s | 135.4 s |

The 7B model does not fit comfortably in 8 GB next to the OS, so it swaps and runs about 5.6x slower. One review needs
about 12 LLM calls, so 7B would take around 27 minutes per paper and the D2 evaluation (8 papers × 2 configurations × 3
seeds) would take more than 20 hours.

**Decision:** `qwen2.5:3b` is the default for development, the D2 evaluation and the live demo, at $0 API cost. The
LLM client is provider-agnostic (`LLM_PROVIDER=anthropic|openai`). The plan to W12 adds a hosted small model run for
the final report.
