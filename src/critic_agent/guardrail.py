"""Prompt-injection guardrail applied at every untrusted entry point: PDF text, user request, tool output, memory.

Defence in depth (see docs/threat_model.md):
  L1  PDF parser drops text a human cannot see (white/low-contrast, tiny, covered by a shape).
  L2  This module: normalise the text, then redact instruction-like sentences (rules below).
  L3  Prompts spotlight untrusted content inside <untrusted> tags (prompts.py).
  L4  Citations are verified against the PDF; tools are permission-gated (evaluator, gateway).

L2 is a deterministic filter, not a classifier: it neutralises known attack shapes and is cheap to run everywhere.
Patterns stay narrow and *imperative-gated* (the verb must open a sentence or clause) so that descriptive text such as
"Models that ignore the formatting instructions score zero" passes. Measure with `python -m critic_agent.injection_eval`.
"""

from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import NamedTuple

REDACTION = "[REDACTED: suspected injected instruction]"

# ------------------------------------------------------------------ normalisation (detection views)

# Cyrillic / Greek letters that look like Latin ones (homoglyph attack)
_HOMOGLYPHS = str.maketrans(
    "аеорсухіјѕԁɡѵοανικρτυΑΒΕΖΗΙΚΜΝΟΡΤΥΧАВЕЗНІКМНОРСТУХ",
    "aeopcyxijsdgvoanikrtuABEZHIKMNOPTYXABEZHIKMHOPCTYX",
)
_LEET = str.maketrans("013457@$", "oieastas")
_LEET_WORD = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b")
_SPACED = re.compile(r"\b\w(?: {1,2}\w){5,}\b")
_TAG_CHARS = re.compile("[\U000e0020-\U000e007e]")
_B64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}")


def _strip_invisible(text: str) -> str:
    """Remove zero-width, bidi and other format characters (category Cf) that hide or split words."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def _decode_tags(text: str) -> str:
    """Unicode 'tag' characters render as nothing but spell ASCII (ASCII smuggling): make them visible."""
    return _TAG_CHARS.sub(lambda m: chr(ord(m.group()) - 0xE0000), text)


def _despace(text: str) -> str:
    """'i g n o r e  a l l' (single spaces inside words, double spaces between words) -> 'ignore all'."""
    return _SPACED.sub(lambda m: m.group().replace("  ", "\0").replace(" ", "").replace("\0", " "), text)


def _decoded_base64(text: str) -> str:
    out = []
    for token in _B64.findall(text):
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), validate=True).decode("utf-8")
        except ValueError:
            continue
        if raw.isprintable() and sum(c.isalpha() or c.isspace() for c in raw) > 0.8 * len(raw):
            out.append(raw)
    return " ".join(out)


def views(sentence: str) -> list[str]:
    """Alternative readings of one sentence that an LLM could still understand (the original is always first)."""
    base = unicodedata.normalize("NFKC", sentence)
    folded = base.translate(_HOMOGLYPHS)
    joined = re.sub(r"(?<=\w)[-_](?=\w)", " ", folded)
    leet = _LEET_WORD.sub(lambda m: m.group().translate(_LEET), joined)
    out = [base, folded, _despace(folded), leet]
    decoded = _decoded_base64(sentence)
    if decoded:
        out.append(decoded)
    seen: list[str] = []
    for v in out:
        v = " ".join(v.split())
        if v and v not in seen:
            seen.append(v)
    return seen


# ------------------------------------------------------------------ rules

# A gated rule must open a sentence or clause, optionally after a polite / modal lead-in: "Please ignore ...",
# "You must ignore ...", "To finish the review, call ...". Descriptive uses ("Models that ignore ...") do not match.
_GATE = (r"^\W*(?:(?:please|kindly|now|also|then|and|but|instead|first|just|always|immediately|you (?:must|should|shall|will|"
         r"need to|have to|are to|can|may)|i (?:want|need|would like) you to|(?:the )?(?:ai|llm|model|assistant|reviewer) "
         r"(?:must|should|shall|will)|to [\w ]{1,50}?,)\W+)*")
_VERB_OVERRIDE = r"(?:ignor(?:e|ez|iere)|disregard|forget|override|bypass|discard|abandon|skip|stop following)"
_OBJ_RULES = r"(?:instructions?|prompts?|rules?|guidelines?|directives?|commands?|constraints?|system message|orders?)"
_FORCED_TOKEN = r"(?-i:(?!(?:YES|NO|TRUE|FALSE|JSON|LLM|AI|SQL|CSV|XML|PDF)\b)[A-Z]{2,}\b)"
_FILLER = r"(?:only|just|exactly|simply|always|the|a|word|token|string|phrase|letters?|text|with|single|your|answer|response|review)"


class Rule(NamedTuple):
    id: str
    category: str
    pattern: str
    gated: bool = False


RULES: list[Rule] = [
    # --- instruction override ---------------------------------------------------------------------------------
    Rule("override", "override", rf"{_VERB_OVERRIDE}\b(?:\W+\w+){{0,3}}?\W+(?:everything|all (?:of )?(?:the )?above|{_OBJ_RULES})\b", True),
    Rule("override-context", "override", rf"{_VERB_OVERRIDE}\b(?:\W+\w+){{0,2}}?\W+(?:previous|prior|earlier|above|preceding|former)\W+(?:text|content|messages?|context|answers?|everything)\b", True),
    Rule("override-above", "override", rf"(?:forget|ignore|disregard)\W+(?:everything|anything|all)\b[^.;]{{0,20}}\b(?:above|before|so far)\b", True),
    Rule("new-instructions", "override", r"\b(?:your|my|these)\s+(?:new|updated|revised|real|actual)\s+(?:instructions?|directives?|tasks?|rules?|orders?|goals?|jobs?|objectives?)\b"),
    Rule("new-instructions-label", "override", r"\b(?:new|updated|revised|override)\s+(?:instructions?|directives?|rules?|orders?|task)\s*:"),
    Rule("from-now-on", "override", r"\bfrom\s+(?:now|this point|here)\s+on(?:wards?)?\b[^.;]{0,50}\b(?:you|always|never|must|only|give|rate|write|answer|ignore)\b"),
    Rule("outdated", "override", rf"\b{_OBJ_RULES}\b[^.;]{{0,40}}\b(?:outdated|no longer (?:apply|valid|relevant)|(?:are|is) void|have been (?:replaced|superseded|revoked))\b"),
    Rule("instead-of-above", "override", r"\binstead of (?:the )?(?:rules|ones|instructions|guidelines)\b[^.;]{0,20}\b(?:above|before|earlier)\b"),
    Rule("follow-new", "override", r"\b(?:follow|here are|obey)\s+(?:the |these )?new\s+(?:instructions|rules)\b"),
    # --- role / persona hijack --------------------------------------------------------------------------------
    Rule("you-are-now", "role-hijack", r"\byou are now\b|\bfrom now on,? you (?:are|will be)\b"),
    Rule("jailbreak-mode", "role-hijack", r"(?:\byou are (?:now )?in\b|\b(?:enable|enter|activate|switch to)\b)[^.;]{0,20}\b(?:developer|admin|god|jailbreak|debug|dan) mode\b|(?-i:\bDAN\b)"),
    Rule("pretend", "role-hijack", r"\b(?:pretend|imagine|roleplay|role-play)\b[^.;]{0,15}\b(?:you are|you're|to be|that you|as)\b", True),
    Rule("unrestricted", "role-hijack", r"\b(?:unrestricted|unfiltered|uncensored)\s+(?:ai|model|assistant|llm)\b|\bno\s+(?:safety\s+)?(?:rules|restrictions|filters|limits)\b"),
    Rule("act-as-author", "role-hijack", r"\bact as\b[^.;]{0,20}\b(?:first |lead |corresponding )?authors?\b", True),
    Rule("hypothetical", "role-hijack", r"\bhypothetical (?:world|scenario|situation)\b[^.;]{0,50}\b(?:no rules|no restrictions|you had no|without rules)\b"),
    # --- fake templates / structure injection -----------------------------------------------------------------
    Rule("chat-tokens", "fake-template", r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext)\s*\|>|\[/?INST\]|<<\s*/?SYS\s*>>|<\s*/?\s*system\s*>|\[\s*system\s*\]|</?untrusted>"),
    Rule("md-header-role", "fake-template", r"#{2,}\s*(?:system|new instructions?|instruction override)\b"),
    Rule("json-forgery", "fake-template", r"\b(?:no_issue|evidence_ids|accept_score)\b"),
    # --- review manipulation (the attacker is the author of the paper) ----------------------------------------
    Rule("address-ai", "review-manipulation", r"\b(?:dear|attention|note to(?: the)?|hey|hello)\s+(?:ai|llm|language model|assistant|automated reviewer|ai reviewer|reviewer model)\b|\bto (?:any|all) (?:ai|llm|automated) (?:reviewer|assistant|system|agent)s?\b"),
    Rule("ai-reviewers-should", "review-manipulation", r"\b(?:ai|llm|automated|language[- ]model)\s+(?:reviewers?|critics?|graders?|evaluators?)\b[^.;]{0,25}\b(?:should|must|shall|are (?:asked|requested|instructed|expected|required) to|please|kindly)\b"),
    Rule("if-you-are-llm", "review-manipulation", r"\bif you are (?:an? )?(?:ai|llm|language model|assistant|chatbot|automated)\b|\bas an? (?:ai|llm) (?:reviewer|assistant)\b[^.;]{0,20}\byou (?:must|should|will)\b"),
    Rule("positive-review", "review-manipulation", r"\b(?:give|assign|award|write|provide|produce|return)\b[^.;]{0,30}\b(?:positive|favou?rable|glowing|excellent|perfect|highest|top|maximum|full)\s+(?:possible\s+)?(?:review|score|rating|evaluation|assessment|marks|grade)\b", True),
    Rule("recommend-accept", "review-manipulation", r"\brecommend\s+(?:acceptance|accepting|to accept|accept)\b", True),
    Rule("deserves-top", "review-manipulation", r"\b(?:deserves?|merits?)\b[^.;]{0,30}\b(?:highest|top|maximum|full|perfect)\b"),
    Rule("rate-perfect", "review-manipulation", r"\b(?:rate|score|grade|rating|score of)\b[^.;]{0,20}\b(?:this|the) (?:paper|work|submission)\b[^.;]{0,25}\b(?:10|10/10|perfect|flawless|maximum|highest|full marks)\b"),
    Rule("always-score", "review-manipulation", r"\balways\s+(?:give|assign|rate|score|answer|state|claim|report|say|respond)\b"),
    Rule("flawless-paper", "review-manipulation", r"\b(?:this|the) (?:paper|work|submission) (?:is|has|are|have)\s+(?:(?:flawless|perfect)\b(?!\s+for)|(?:no|zero) (?:known )?(?:weaknesses|flaws|limitations|issues))"),
    Rule("suppress", "review-manipulation", r"\b(?:do not|don't|never|avoid|refrain from|omit|leave out|skip|suppress|hide|conceal)\b[^.;]{0,60}\b(?:weakness(?:es)?|limitations?|flaws?|contamination|negatives?|negative points?|criticisms?|drawbacks?|shortcomings?|concerns?)\b", True),
    Rule("claim-that", "review-manipulation", r"\b(?:state|claim|say|write|assert|conclude|declare|report)\s+that\s+(?:this paper|the paper|this work|all baselines|the (?:method|approach|evaluation|results|authors)|every|no)\b", True),
    Rule("manipulate-axes", "review-manipulation", r"\b(?:skip|ignore|omit|mark|set|treat|accept)\b[^.;]{0,40}\b(?:axis|axes|critiques?|critic|evaluator|rating|score)\b", True),
    Rule("forced-output", "review-manipulation", rf"\b(?:write|say|output|print|reply|respond|answer|return)(?:\W+{_FILLER})*\W+{_FORCED_TOKEN}", True),
    Rule("forced-output-quote", "review-manipulation", r"\b(?:begin|start|end|finish)\s+(?:your|the)\s+(?:answer|response|reply|output|review|critique)\s+with\b", True),
    Rule("pipeline-agents", "review-manipulation", r"\b(?:the )?(?:evaluator|critics?|orchestrator|synthesizer|summarizer)\b\W+(?:must|should|shall|will)\b"),
    # --- system-prompt / secret leakage -----------------------------------------------------------------------
    Rule("leak-verb", "prompt-leak", r"\b(?:print|repeat|output|show|reveal|write out|list|display|copy|echo|recite|disclose|share|leak|tell me)\b[^.;]{0,30}\b(?:(?:your|the|all|any|my|these)\s+)*(?:system\s+prompt|hidden\s+prompt|initial\s+prompt|instructions?|prompt|rules|guidelines|directives)\b", True),
    Rule("leak-above", "prompt-leak", r"\b(?:repeat|print|output|show|write out|copy|echo)\b[^.;]{0,25}\b(?:text|words|everything|content|messages?|instructions?)\b[^.;]{0,15}\b(?:above|before this|so far|you were given)\b", True),
    Rule("leak-question", "prompt-leak", r"\b(?:what|which)\s+(?:is|are|were)\s+your\s+(?:system\s+|initial\s+|hidden\s+)?(?:prompt|instructions|rules)\b|\byour\s+(?:system|hidden|initial)\s+prompt\b"),
    Rule("verbatim", "prompt-leak", r"\b(?:print|repeat|output|show|write)\b[^.;]{0,20}\bverbatim\b", True),
    Rule("internal-id", "prompt-leak", r"\binternal\s+(?:reference\s+)?(?:id|identifier|key|token)s?\b|zx-canary"),
    # --- tool abuse -------------------------------------------------------------------------------------------
    Rule("tool-names", "tool-abuse", r"\b(?:execute_code_in_sandbox|save_annotation|delete_annotation|publish_review|search_paper|pdf_ingest|artifact_check|graph_query|get_sentences)\b"),
    Rule("tool-write", "tool-abuse", r"\b(?:use|call|invoke|run|execute)\b[^.;]{0,30}\b(?:save|publish|delete|submit|execute)\b[^.;]{0,15}\b(?:annotations?|reviews?|code|scripts?)\b", True),
    # --- exfiltration -----------------------------------------------------------------------------------------
    Rule("md-image", "exfiltration", r"!\[[^\]]*\]\(\s*https?://[^)\s]*\?[^)\s]*(?:prompt|system|secret|key|token|data|q|query|text|content|info)="),
    Rule("send-out", "exfiltration", r"\b(?:send|post|upload|transmit|forward|email|append|attach)\b[^.;\n]{0,60}(?:\bto\s+https?://|~/|/etc/|\.ssh|id_rsa|\bpasswords?\b|\bapi[_ ]?keys?\b|credentials)", True),
    # --- authority / social engineering -----------------------------------------------------------------------
    Rule("authority-from", "authority", r"\b(?:message|notice|note|announcement|directive|order|update|instruction)s?\s+from\s+(?:the\s+)?(?:course\s+)?(?:instructor|professor|administrators?|admins?|program chairs?|chairs?|organi[sz]ers?|developers?|committee|system|anthropic|openai|moderators?)\b|\b(?:urgent|important)\b[^.;]{0,15}\bfrom\s+(?:the\s+)?(?:program chairs?|instructor|administrators?|committee)\b"),
    Rule("authority-label", "authority", r"\b(?:administrator|admin|instructor|committee)\s+(?:notice|message|note|override|command|order)\b"),
    # --- memory / lesson poisoning ----------------------------------------------------------------------------
    Rule("memory-lesson", "memory-poisoning", r"\blessons?\b\W{0,12}[^.]{0,60}\b(?:always|never)\b|\bare (?:always|automatically) (?:accepted|approved|rated|scored)\b"),
    Rule("next-time", "memory-poisoning", r"\b(?:next time|in future|in the future|henceforth)\b[^.;]{0,40}\b(?:praise|accept|approve|agree|ignore|skip|always|never)\b"),
    Rule("never-question", "memory-poisoning", r"\bnever\s+(?:question|criticiz|doubt|challenge)\w*\b"),
    # --- encoded payloads -------------------------------------------------------------------------------------
    Rule("decode-follow", "obfuscation", r"\b(?:decode|decrypt|deobfuscate)\b[^.;]{0,30}\b(?:follow|execute|obey|run)\b"),
    # --- other languages: override verb + instruction noun ----------------------------------------------------
    Rule("vi-override", "multilingual", r"(?:bỏ qua|phớt lờ|quên đi)\W+(?:\w+\W+){0,4}?(?:hướng dẫn|chỉ thị|chỉ dẫn|quy tắc|lệnh)"),
    Rule("fr-override", "multilingual", r"\bignor(?:ez|er|e)\s+(?:toutes\s+)?(?:les|vos|ces|mes)\b(?:\W+\w+){0,2}?\W+(?:instructions?|consignes?|règles?|directives?)\b"),
    Rule("es-override", "multilingual", r"\bignor(?:a|ar)\s+(?:todas\s+)?(?:las|tus|sus)\b(?:\W+\w+){0,2}?\W+(?:instrucciones|reglas|directrices)\b"),
    Rule("de-override", "multilingual", r"\bignorier(?:e|en|t)?\b\W+(?:\w+\W+){0,3}?(?:anweisungen|regeln|instruktionen|vorgaben)\b"),
    Rule("zh-override", "multilingual", r"(?:忽略|无视|忘记|忽视|忽略掉)[^。！？\n]{0,12}(?:指令|指示|规则|提示|说明|要求)"),
    Rule("ja-override", "multilingual", r"(?:指示|命令|ルール|指令)[^。！？\n]{0,12}(?:無視|忘れ)"),
]

_FLAGS = re.IGNORECASE
_COMPILED = [(r, re.compile(_GATE + r"(?:" + r.pattern + r")" if r.gated else r.pattern, _FLAGS)) for r in RULES]
INJECTION_PATTERNS = [r.pattern for r in RULES]  # kept for scripts that list the patterns

_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n\s*\n")
_CLAUSE = re.compile(r"[;:,()\[\]{}\"“”‘’|—–]\s*|\s-\s")


@dataclass
class Flag:
    source: str  # pdf | pdf-hidden | user | tool:<name> | memory
    page: int | None
    snippet: str
    rule: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def match_rule(sentence: str) -> str | None:
    """Id of the first rule that fires on the sentence (checking every normalised view), else None."""
    for view in views(sentence):
        clauses = [view] + [c for c in _CLAUSE.split(view) if c.strip()]
        for rule, pattern in _COMPILED:
            if rule.gated:
                if any(pattern.match(c) for c in clauses):
                    return rule.id
            elif pattern.search(view):
                return rule.id
    return None


def _spans(text: str):
    """(start, end) of each sentence in `text`; single line breaks (PDF line wraps) stay inside a sentence."""
    start = 0
    for m in _SENTENCE.finditer(text):
        yield start, m.start()
        start = m.end()
    yield start, len(text)


def find_injections(text: str) -> list[str]:
    return [text[a:b] for a, b in _spans(_decode_tags(text)) if text[a:b].strip() and match_rule(text[a:b])]


def sanitize(text: str, source: str, page: int | None = None) -> tuple[str, list[Flag]]:
    """Invisible characters are stripped, instruction-like sentences are replaced by REDACTION, flags say why."""
    text = _strip_invisible(_decode_tags(text))
    flags: list[Flag] = []
    out, last = [], 0
    for a, b in _spans(text):
        rule = match_rule(text[a:b]) if text[a:b].strip() else None
        if rule:
            out += [text[last:a], REDACTION]
            last = b
            flags.append(Flag(source, page, " ".join(text[a:b].split())[:160], rule))
    out.append(text[last:])
    return "".join(out), flags


def clean_paper(paper) -> tuple[list[str], list[dict]]:
    """Defence for one parsed PDF: drop hidden payload blocks (already split by the parser), then redact
    instruction-like sentences page by page. Returns the clean pages and the flags as dicts."""
    flags = [{"source": "pdf-hidden", "page": h.page, "snippet": h.text[:160], "rule": h.reason} for h in paper.hidden_spans]
    pages = []
    for page_no, text in enumerate(paper.visible_pages, start=1):
        clean, page_flags = sanitize(text, "pdf", page_no)
        pages.append(clean)
        flags += [f.to_dict() for f in page_flags]
    return pages, flags
