"""Measure prompt/generation speed of local Ollama models on a paper-sized prompt (used to choose the D2 model)."""

import json
import sys
import time

import httpx
import pymupdf

MODELS = sys.argv[1:] or ["qwen2.5:3b", "qwen2.5:7b"]
text = " ".join(pymupdf.open("data/papers/autocoderover.pdf")[6].get_text().split())[:9000]
prompt = (
    "From the passage below, list the 3 most important weaknesses in the experimental fairness of the paper. "
    'Return JSON {"weaknesses": [{"claim": str, "evidence": str}]}.\n\n' + text
)
for model in MODELS:
    start = time.perf_counter()
    r = httpx.post("http://localhost:11434/api/generate", timeout=900, json={
        "model": model, "prompt": prompt, "format": "json", "stream": False,
        "options": {"num_ctx": 8192, "temperature": 0.2, "seed": 1}})
    d = r.json()
    print(json.dumps({
        "model": model, "wall_s": round(time.perf_counter() - start, 1),
        "prompt_tokens": d.get("prompt_eval_count"), "prompt_tok_s": round(d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9), 1),
        "gen_tokens": d.get("eval_count"), "gen_tok_s": round(d["eval_count"] / (d["eval_duration"] / 1e9), 1),
        "sample": d["response"][:300]}, ensure_ascii=False))
