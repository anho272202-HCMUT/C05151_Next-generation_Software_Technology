"""Render a review (full or baseline) as a Markdown sheet."""

from __future__ import annotations

AXIS_TITLES = {
    "assumptions": "Assumptions",
    "fairness": "Fairness of comparison",
    "data_reliability": "Data reliability",
    "generalizability": "Generalizability",
    "reproducibility": "Reproducibility",
}


def render_markdown(final: dict) -> str:
    lines = [f"# Critic-Agent review: {final.get('title') or final['paper_id']}", ""]
    if final.get("mode") == "full":
        lines += [f"Accepted critiques: **{final['accepted']}/5** · rounds: {final['rounds']} · "
                  f"attempts: {final['attempts']}", ""]
    for c in final["critiques"]:
        status = "accepted" if c.get("accepted") else ("grounded" if c.get("grounded") else "NOT VERIFIED")
        score = f", score {c['score']:.0f}/10" if c.get("score") is not None else ""
        lines.append(f"## {AXIS_TITLES.get(c['axis'], c['axis'])} ({status}{score})")
        lines.append(f"**Weakness:** {c['weakness']}")
        for q in c.get("quotes", []):
            lines.append(f"> [{q['id']}, p.{q['page']}] {q['text']}")
        if c.get("question"):
            lines.append(f"**Debate question:** {c['question']}")
        lines.append("")
    if final.get("debate_questions"):
        lines.append("## Top debate questions")
        lines += [f"{i}. {q}" for i, q in enumerate(final["debate_questions"], 1)] + [""]
    flags = final.get("guardrail_flags", []) + final.get("tool_flags", [])
    if flags:
        lines.append("## Security notices")
        lines += [f"- Neutralised `{f['source']}` content (page {f.get('page')}): `{f['snippet'][:100]}`" for f in flags]
    return "\n".join(lines) + "\n"
