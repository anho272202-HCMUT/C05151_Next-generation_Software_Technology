"""Measure how well the prompt-injection defences neutralise a fixed attack set and how often they hurt benign text.

uv run python -m critic_agent.injection_eval [--corpus] [--out results/injection_eval.json] [--export-web]

Attacks come from eval/injections.json. An attack is NEUTRALISED when its `marker` (a distinctive part of the malicious
instruction) no longer reaches the LLM after the defence. Each attack is run through four configurations so the
contribution of every layer is visible:

  none       no defence (what a naive extractor feeds the LLM)
  parser     PDF hidden-text removal only
  sanitizer  normalisation + instruction redaction only
  full       parser + sanitizer, exactly what pdf_ingest / the gateway / the user-request path run

Only deterministic layers are measured here. Whether a given LLM actually obeys an injection that slips through is a
separate question that needs a real model.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import pymupdf

from .config import PROJECT_ROOT
from .guardrail import _decode_tags, clean_paper, sanitize
from .pdf_parser import parse_pdf, split_sentences

LAYERS = ("none", "parser", "sanitizer", "full")
BODY = ("We evaluate our agent on SWE-bench Lite with GPT-4 and report the percentage of resolved instances. "
        "All baselines use the same model and the same cost budget. Results are averaged over three runs.")

# how a hidden payload is drawn: (font size, RGB colour, extra)
HIDE_STYLES: dict[str, dict] = {
    "white": {"size": 8, "color": (1, 1, 1)},
    "near_white": {"size": 8, "color": (0.99, 0.99, 0.99)},
    "light_gray": {"size": 8, "color": (0.97, 0.97, 0.97)},
    "tiny_1pt": {"size": 1, "color": (0, 0, 0)},
    "tiny_2pt": {"size": 2, "color": (0, 0, 0)},
    "tiny_3pt": {"size": 3, "color": (0, 0, 0)},
    "covered": {"size": 8, "color": (0, 0, 0), "cover": True},
    "off_page": {"size": 8, "color": (0, 0, 0), "off_page": True},
}


def build_attack_pdf(path: Path, payload: str, hide: str) -> Path:
    """A one-page PDF with a normal paragraph and `payload` hidden in the given style."""
    style = HIDE_STYLES[hide]
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 150), BODY, fontsize=10)
    if style.get("off_page"):
        page.insert_text((50, -30), payload, fontsize=style["size"], color=style["color"])
    else:
        rect = pymupdf.Rect(50, 700, 550, 780)
        page.insert_textbox(rect, payload, fontsize=style["size"], color=style["color"])
        if style.get("cover"):
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
    doc.save(path)
    return path


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def reaches_llm(attack: dict, layer: str, workdir: Path) -> bool:
    """True if the attack marker is still present in what the LLM would read under this configuration."""
    marker = _norm(attack["marker"])
    if attack["channel"] == "text":
        text = sanitize(attack["text"], "text")[0] if layer in ("sanitizer", "full") else attack["text"]
        return marker in _norm(_decode_tags(text))  # LLMs read Unicode tag characters as ASCII
    pdf = build_attack_pdf(workdir / f"{attack['id']}.pdf", attack["text"], attack["hide"])
    paper = parse_pdf(pdf)
    if layer == "full":
        pages, _ = clean_paper(paper)
    elif layer == "parser":
        pages = paper.visible_pages
    elif layer == "sanitizer":
        pages = [sanitize(p, "pdf", i)[0] for i, p in enumerate(paper.raw_pages, start=1)]
    else:
        pages = paper.raw_pages
    return marker in _norm(" ".join(s.text for s in split_sentences(pages)))


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 3) if total else 0.0


def evaluate_attacks(attacks: list[dict]) -> dict:
    per_attack, by_layer = [], {layer: defaultdict(lambda: [0, 0]) for layer in LAYERS}
    with tempfile.TemporaryDirectory() as tmp:
        for atk in attacks:
            row = {"id": atk["id"], "technique": atk["technique"], "channel": atk["channel"], "split": atk["split"],
                   "text": atk["text"][:200], "neutralised": {}}
            if atk["channel"] == "pdf-hidden":
                row["hide"] = atk["hide"]
            else:  # what the sanitiser does with it, for the web examples
                clean, flags = sanitize(atk["text"], "text")
                row["rule"], row["after"] = (flags[0].rule if flags else None), clean[:200]
            for layer in LAYERS:
                done = not reaches_llm(atk, layer, Path(tmp))
                row["neutralised"][layer] = done
                for key in ("all", atk["technique"], f"split:{atk['split']}"):
                    by_layer[layer][key][0] += done
                    by_layer[layer][key][1] += 1
            per_attack.append(row)
    summary = {layer: {k: {"neutralised": v[0], "total": v[1], "rate": _rate(*v)} for k, v in groups.items()}
               for layer, groups in by_layer.items()}
    return {"summary": summary, "attacks": per_attack}


def evaluate_benign(sentences: list[str]) -> dict:
    flagged = [s for s in sentences if sanitize(s, "text")[1]]
    return {"total": len(sentences), "flagged": len(flagged), "rate": _rate(len(flagged), len(sentences)),
            "false_positives": flagged}


def evaluate_corpus(papers_dir: Path) -> dict | None:
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        return None
    rows, total_sentences, total_flags = [], 0, 0
    for pdf in pdfs:
        paper = parse_pdf(pdf)
        pages, flags = clean_paper(paper)
        n = len(split_sentences(paper.visible_pages))
        redacted = [f for f in flags if f["source"] == "pdf"]
        rows.append({"paper": pdf.name, "sentences": n, "hidden_blocks": len(paper.hidden_spans),
                     "redacted_sentences": len(redacted), "snippets": [f["snippet"] for f in redacted][:5]})
        total_sentences += n
        total_flags += len(redacted) + len(paper.hidden_spans)
    return {"papers": rows, "sentences": total_sentences, "flags": total_flags, "rate": _rate(total_flags, total_sentences)}


def run(corpus: bool = True) -> dict:
    attacks = json.loads((PROJECT_ROOT / "eval" / "injections.json").read_text())["attacks"]
    benign = json.loads((PROJECT_ROOT / "eval" / "benign.json").read_text())["sentences"]
    result = {"n_attacks": len(attacks), **evaluate_attacks(attacks), "benign": evaluate_benign(benign)}
    if corpus:
        result["corpus"] = evaluate_corpus(PROJECT_ROOT / "data" / "papers")
    return result


def format_report(result: dict) -> str:
    s = result["summary"]
    techniques = sorted({a["technique"] for a in result["attacks"]})
    lines = [f"Attacks: {result['n_attacks']}   (neutralised = marker no longer reaches the LLM)", "",
             f"{'':24s}" + "".join(f"{layer:>11s}" for layer in LAYERS)]

    def row(label, key):
        cells = "".join(f"{s[layer][key]['neutralised']:>5d}/{s[layer][key]['total']:<5d}" for layer in LAYERS)
        lines.append(f"{label:24s}{cells}")

    row("ALL", "all")
    row("  dev", "split:dev")
    row("  heldout", "split:heldout")
    for t in techniques:
        row(f"  {t}", t)
    b = result["benign"]
    lines += ["", f"False positives on hard benign sentences: {b['flagged']}/{b['total']}"]
    lines += [f"   - {t}" for t in b["false_positives"]]
    c = result.get("corpus")
    if c:
        lines += ["", f"Corpus false positives: {c['flags']} flags in {c['sentences']} sentences over {len(c['papers'])} papers"]
        lines += [f"   - {p['paper']}: {p['redacted_sentences']} redacted, {p['hidden_blocks']} hidden blocks {p['snippets']}"
                  for p in c["papers"] if p["redacted_sentences"] or p["hidden_blocks"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-corpus", action="store_true", help="skip the false-positive scan of data/papers/")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "injection_eval.json")
    ap.add_argument("--export-web", action="store_true", help="also write web/data/injection_eval.json")
    args = ap.parse_args()
    result = run(corpus=not args.no_corpus)
    print(format_report(result))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if args.export_web:
        web = PROJECT_ROOT / "web" / "data" / "injection_eval.json"
        before = PROJECT_ROOT / "results" / "injection_eval_before.json"
        if before.exists():  # the guardrail of D2, measured with the same harness, for the before/after comparison
            old = json.loads(before.read_text())
            result["before"] = {"summary": old["summary"]["full"], "benign": {k: old["benign"][k] for k in ("total", "flagged", "rate")},
                                "per_attack": {a["id"]: a["neutralised"]["full"] for a in old["attacks"]}}
        web.write_text(json.dumps(result, ensure_ascii=False))
        print(f"\nweb data -> {web}")


if __name__ == "__main__":
    main()
