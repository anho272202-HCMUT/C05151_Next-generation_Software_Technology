"""Render a review (full or baseline) as a Markdown sheet."""

from __future__ import annotations

AXIS_TITLES = {
    "assumptions": "Assumptions",
    "fairness": "Fairness of comparison",
    "data_reliability": "Data reliability",
    "generalizability": "Generalizability",
    "reproducibility": "Reproducibility",
}


def response_label(r: dict) -> str:
    """Outcome of the author-response check of one accepted weakness."""
    if not r.get("accurate", True):
        return "refuted: the critique misreads the paper"
    parts = [{"fully": "fully addressed", "partly": "partly addressed"}.get(r.get("addressed"), "not addressed")]
    if r.get("acknowledged"):
        parts.append("acknowledged by the authors")
    return ", ".join(parts)


def render_markdown(final: dict) -> str:
    lines = [f"# Critic-Agent review: {final.get('title') or final['paper_id']}", ""]
    if final.get("mode") == "full":
        lines += [f"Accepted critiques: **{final['accepted']}/5** · rounds: {final['rounds']} · "
                  f"attempts: {final['attempts']}", ""]
    lines += _assessment(final.get("assessment") or {})
    for c in final["critiques"]:
        status = ("no significant weakness found" if c.get("no_issue") else
                  "accepted" if c.get("accepted") else ("grounded" if c.get("grounded") else "NOT VERIFIED"))
        score = f", critique score {c['score']:.0f}/10" if c.get("score") is not None and not c.get("no_issue") else ""
        lines.append(f"## {AXIS_TITLES.get(c['axis'], c['axis'])} ({status}{score})")
        lines.append(f"**Weakness:** {c['weakness']}")
        for q in c.get("quotes", []):
            lines.append(f"> [{q['id']}, p.{q['page']}] {q['text']}")
        if c.get("response"):
            lines.append(f"**Against the rest of the paper:** {response_label(c['response'])}"
                         + (f" ({c['response']['note']})" if c["response"].get("note") else ""))
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


def _assessment(a: dict) -> list[str]:
    """Paper score from the Meta-reviewer. The per-axis scores below grade the critiques, not the paper."""
    if not a.get("overall"):
        return []
    lines = ["## Paper assessment (Meta-reviewer)", f"**Overall: {a['overall']:g}/10**"
             + (f" · confidence {a['confidence']:g}/5" if a.get("confidence") else "")]
    if a.get("dimensions"):
        lines.append(" · ".join(f"{d} {v:g}" for d, v in a["dimensions"].items()))
    lines += [f"- Strength: {x}" for x in a.get("strengths", [])]
    if a.get("justification"):
        lines.append(f"\n{a['justification']}")
    return lines + [""]
