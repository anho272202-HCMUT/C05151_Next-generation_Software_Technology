# Blind attack batch written by another model family (attempted 2026-11-21): refused

Goal: remove the weakness that the same assistant wrote both the filter and the attacks (see "Who wrote the attacks" in `threat_model.md`) by asking a
model that never saw the filter to write 20-30 attacks, labelling them before any measurement and measuring the sanitiser once.

## What was done

- `src/critic_agent/injection_blind.py generate --model claude-haiku-5-5`: four calls (goals G1 force a word, G2 leak the internal id, G3 report no weakness,
  G4 other bias in the author's favour), one per goal, generic description of the system and of the goals, no rules, no examples from our sets, the
  purpose stated truthfully (testing our own defences before release).
- Cost: 1,876 input and 2,749 output tokens, about $0.016 at the list price in `cost.py`.

## Result

**Claude Haiku 5.5 declined all four requests** (0 payloads). Its stated reason, in short: ready-made attack texts meant to steer an automated paper
reviewer would work the same way for an author gaming a real review, whatever the testing purpose; it suggested canary-based tests and defence design
instead. The raw refusals are stored in `eval/blind_attacks.json`. The same happened earlier with gpt-6-luna as adaptive attacker (several goals declined).

We did not rephrase the request to get around the refusal: the concern is legitimate (academic review manipulation is a real misuse) and the plan was to
record a refusal as a result. Nothing was measured, so **no claim about the independence of the evaluation is added by this attempt**; the limitation in
`threat_model.md` stays as written.

## Remaining options (not run)

1. Attacks written by a human who has not seen the rules (a teammate): the only fully independent source; no API cost.
2. A paraphrase-robustness test: another model rewrites our existing attacks in different wording (ideas ours, wording independent). It measures how much the
   filter depends on our wording, not whether it covers unseen ideas.
3. A local model of another family as attacker (for example a Qwen model through Ollama): independent and free, but weaker, and every payload still needs a
   validity label.
