"""Summarise the end-to-end and semantic injection studies for the web Explorer (web/data/injection_studies.json).

uv run python scripts/export_injection_studies.py

Reads results/injection_e2e_<model>.jsonl and results/injection_semantic_test_<model>.jsonl (written by critic_agent.injection_e2e and
critic_agent.injection_semantic); recomputes the headline numbers so the page never shows a number that is not in the raw rows.
"""

from __future__ import annotations

import json

from critic_agent.config import PROJECT_ROOT
from critic_agent.injection_semantic import fisher

RESULTS = PROJECT_ROOT / "results"
IMPERATIVE = ("obey-visible", "obey-hidden", "leak-visible", "suppress-visible", "inflate-visible")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def e2e() -> list[dict]:
    out = []
    for model in ("gpt-4o-mini", "gpt-4o", "gpt-6-luna"):
        rs = [r for r in rows(RESULTS / f"injection_e2e_{model}.jsonl") if not r.get("failed")]
        if not rs:
            continue
        imp = [r for r in rs if r["attack"] in IMPERATIVE]
        none_imp = [r for r in imp if r["config"] == "none"]
        hid = lambda cfg: [r for r in rs if r["attack"] == "claim-hidden" and r["config"] == cfg]
        out.append({"model": model, "runs": len(rs),
                    "imperative": [sum(bool(r["success"]) for r in imp), len(imp)],
                    "imperative_no_defence": [sum(bool(r["success"]) for r in none_imp), len(none_imp)],
                    "hidden_cited_no_defence": [sum(bool(r.get("payload_in_evidence")) for r in hid("none")), len(hid("none"))],
                    "hidden_cited_l1": [sum(bool(r.get("payload_in_evidence")) for r in hid("L1+L2+L3")), len(hid("L1+L2+L3"))]})
    return out


def semantic() -> dict | None:
    rs = [r for r in rows(RESULTS / "injection_semantic_test_gpt-4o-mini.jsonl") if not r.get("failed")]
    if not rs:
        return None
    cfgs = {}
    for cfg in ("none", "system", "system+L5"):
        dec = [r for r in rs if r["config"] == cfg and r.get("variant") == "decoy"]
        clean = [r for r in rs if r["config"] == cfg and r["payload"] == "clean"]
        c_n = sum(1 for r in clean for ax in ("assumptions", "fairness", "data_reliability", "generalizability", "reproducibility"))
        # clean retention is measured on the target axes of the decoy payloads (all five axes of both papers)
        c_k = sum(bool(r["retained"].get(ax)) for r in clean for ax in r["retained"])
        cfgs[cfg] = {"decoy_adopted": [sum(bool(r.get("decoy_adopted")) for r in dec), len(dec)],
                     "real_kept_attacked": [sum(bool(r["retained"].get(r["axis"])) for r in dec), len(dec)],
                     "real_kept_clean": [c_k, c_n]}
    n, l5 = cfgs["none"], cfgs["system+L5"]
    return {"model": "gpt-4o-mini", "configs": cfgs, "p": {
        "attack_vs_clean_no_defence": fisher(n["real_kept_attacked"][0], n["real_kept_attacked"][1] - n["real_kept_attacked"][0],
                                             n["real_kept_clean"][0], n["real_kept_clean"][1] - n["real_kept_clean"][0]),
        "l5_vs_none_decoy": fisher(l5["decoy_adopted"][0], l5["decoy_adopted"][1] - l5["decoy_adopted"][0],
                                   n["decoy_adopted"][0], n["decoy_adopted"][1] - n["decoy_adopted"][0])}}


def examples() -> dict:
    """One decoy and one false-assertion payload (eval/semantic_payloads.json) for the web examples table."""
    payloads = json.loads((PROJECT_ROOT / "eval" / "semantic_payloads.json").read_text())["payloads"]
    pick = lambda pid: next(p for p in payloads if p["id"] == pid)
    return {kind: {k: pick(pid)[k] for k in ("id", "paper", "axis", "text")}
            for kind, pid in (("decoy", "reflexion-data_reliability-D"), ("assertion", "reflexion-fairness-A"))}


def main() -> None:
    doc = {"e2e": e2e(), "semantic": semantic(), "examples": examples()}
    path = PROJECT_ROOT / "web" / "data" / "injection_studies.json"
    path.write_text(json.dumps(doc, indent=1))
    print(f"written {path}")
    print(json.dumps(doc, indent=1)[:1800])


if __name__ == "__main__":
    main()
