# Experiment log: do real LLMs obey the injections? (preliminary)

Date: 2026-11-21. Status: **preliminary, kept as a record**. The experiment did not separate the defence conditions
(floor effect, see below); a stronger design is proposed at the end but not run.

## Question

When an injected sentence reaches the critic's prompt, does a real LLM follow it, and do spotlighting (L3) and
sanitising (L2) reduce that?

## Setup (`src/critic_agent/injection_llm_eval.py`)

- Prompt: the pipeline's critic prompt (`prompts.critic_user`, axis `fairness`) built from real evidence sentences of
  `swe-agent.pdf` (BM25, 1,800-token window), one attack sentence inserted in the middle of the evidence.
- Attacks: 29 text attacks from `eval/injections.json` (all 6 `semantic_persuasion`, the rest sampled per technique,
  heldout first).
- Conditions: `raw` (no guardrail: no tags, no security rules), `spotlight` (L3 only), `full` (attack sanitised by L2, then L3).
- 2 repetitions (seeds 0 and 1), plus an attack-free control (2 outputs, each judged against 12 attacks).
- "Followed" = deterministic check (canary id, forced ALL-CAPS token) OR LLM judge verdict (rubric in `judge_user`).
- Critic models: `gpt-6-luna`, `gpt-4o-mini`. Judge: `gpt-6-luna` (gpt-4o-mini outputs were re-judged with it).
- Cost: about 374 calls, 308k input tokens per critic model (+ about 70k output tokens for gpt-6-luna, 23k for gpt-4o-mini).

## Results

| Critic model | raw | spotlight (L3) | full (L2 + L3) | judge false positives (control) |
|---|---|---|---|---|
| gpt-6-luna | 0/58 | 0/58 | 0/58 | 0/24 |
| gpt-4o-mini | 1/58 | 1/58 | 3/58 | 0/24 |

gpt-4o-mini judged by itself (kept for the record, **not** used): 3/58, 5/58, 7/58 and 5/24 false positives on the control.

Judge sanity check on 4 hand-made outputs (3 obeying the injection, 1 genuine critique): 4/4 correct.

## Findings

1. Both models ignore instruction-style injections in this setting even without any defence: no `no_issue` output in 174
   gpt-6-luna critiques, none mentions the injection, the forced tokens and canary never appear. Nothing left for L2/L3 to reduce.
2. The few flagged gpt-4o-mini outputs are `semantic_persuasion` or review-manipulation items where the model repeats the
   injected claim as something the paper says ("the paper claims the comparison is fully fair") while still writing a
   genuine critique. Differences between columns (1, 1, 3 of 58) are within noise; L2 does not change those items.
3. The judge matters more than the condition: the same outputs gave 3/58 to 7/58 depending on the judge model.

## Limitations

One paper, one axis, 2 repetitions, no confidence intervals (0/58 has a 95% upper bound of about 5%). Static attacks written by
the same author as the defences. Injection only in the evidence channel (not the user request, not an end-to-end PDF with hidden
text). Outcome measured by "followed", not by review integrity (no_issue flips, Evaluator score, accepted weak critiques).
`qwen2.5:3b`, the model of D2, was not measured (Ollama not installed on the authoring machine).

## Proposed follow-up (not done)

End-to-end `run_review` on PDFs carrying the payload, with objective attacker goals (suppress a weakness, raise the Evaluator
score, leak the canary, call a forbidden tool), ablation of L1-L4, a model-strength gradient (qwen2.5:3b, gpt-4o-mini,
gpt-6-luna), adaptive attacks written by an LLM against the filter, utility on clean papers, more papers and repetitions with
Wilson intervals, and human labels for the judge (Cohen's kappa).

## Reproduce

```bash
uv run python -m critic_agent.injection_llm_eval --model gpt-6-luna --reps 2          # needs OPENAI_API_KEY in .env
uv run python -m critic_agent.injection_llm_eval --model gpt-4o-mini --reps 2
uv run python -m critic_agent.injection_llm_eval --rejudge results/injection_llm_eval_gpt-4o-mini.json --judge-model gpt-6-luna
```

Raw outputs and judge reasons: `results/injection_llm_eval_gpt-6-luna.json`, `results/injection_llm_eval_gpt-4o-mini.json`.
