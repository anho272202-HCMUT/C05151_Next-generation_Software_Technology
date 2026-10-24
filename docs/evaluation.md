# Evaluation protocol (D2)

**Task set.** 40 tasks = 8 coding-agent papers × 5 axes (`eval/labels.json`). Each task lists 1–2 known
weaknesses written by the group from the PDF, with the page where the issue is visible. Target for the final
report: ≥ 20 papers (≥ 100 tasks).

**Configurations.**
- *Full*: the multi-agent system (Orchestrator, 5 parallel axis critics, Evaluator, targeted reflection ≤ 2 rounds,
  graph, artifact check, memory).
- *Single-prompt baseline*: one LLM call over the paper's most relevant sentences (≈ 3k tokens), no tools,
  planning, evaluator or reflection, but the same sentence-id citation format.

Both use `qwen2.5:3b` (Ollama, temperature 0.7), seeds 1, 2, 3, and the same per-paper index. Each Full run gets an
empty long-term memory so runs stay independent.

**Correctness.** A task is *solved* when the critique for that axis is
1. **grounded**: every cited sentence id exists in the paper and, for Full, came from the critic's own evidence
   window (deterministic check), **and**
2. **matches** a ground-truth weakness of that task: decided by an LLM judge (qwen2.5:3b, temperature 0) and, as a
   deterministic reference, by keyword matching.

**Judge reliability.** 30 critiques from seed 1, stratified over configuration × axis, are labelled blind by a human
(`eval/human_labels.csv`, instructions in `eval/HUMAN_LABELING.md`). We report accuracy and Cohen's κ of
(a) the evaluation judge and the keyword matcher against human "matches ground truth", and (b) the runtime Evaluator's
accept decision against human "on-axis and supported".

**Reported metrics.** Solved rate (judge / keyword), grounded rate, coverage, per-axis solved rate, outcome
breakdown (solved / grounded-no-match / ungrounded / no critique), latency, LLM calls, tokens and API-equivalent
cost per paper. Mean ± std over the 3 seeds.

**Reproduce.** `uv run python -m critic_agent.evaluation run | judge | sample | agreement | summarize`. All
intermediate artefacts are in `results/` (`eval_runs.jsonl`, `judgments.jsonl`, `summary.json`, `tables/*.tex`),
so the tables can be regenerated without re-running the models.
