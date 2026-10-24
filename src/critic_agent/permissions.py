"""Tool permission model shared by the MCP server and the client-side gateway.

Four levels (instructor's D2 taxonomy): read-only, reversible write, irreversible write, code execution.
Anything above read-only requires an approval token minted by the gateway after a human says yes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from enum import Enum


class Level(str, Enum):
    READ_ONLY = "read-only"
    REVERSIBLE_WRITE = "reversible-write"
    IRREVERSIBLE_WRITE = "irreversible-write"
    CODE_EXECUTION = "code-execution"


@dataclass(frozen=True)
class ToolPolicy:
    level: Level
    agents: frozenset[str]
    needs_approval: bool
    description: str


TOOL_POLICY: dict[str, ToolPolicy] = {
    "pdf_ingest": ToolPolicy(Level.READ_ONLY, frozenset({"ingest"}), False,
                             "Parse + sanitise + index a PDF from data/papers/ (only the server's cache is written)"),
    "search_paper": ToolPolicy(Level.READ_ONLY, frozenset({"summarizer", "orchestrator", "critic"}), False,
                               "Hybrid search over one indexed paper, returns numbered sentences"),
    "get_sentences": ToolPolicy(Level.READ_ONLY, frozenset({"critic", "evaluator", "synthesizer"}), False,
                                "Return verbatim sentences by id (citation verification)"),
    "graph_query": ToolPolicy(Level.READ_ONLY, frozenset({"orchestrator", "critic"}), False,
                              "Neighbours of a paper in the paper graph (<= 20 nodes)"),
    "save_annotation": ToolPolicy(Level.REVERSIBLE_WRITE, frozenset({"user"}), False,
                                  "Append a note to the session file; undone by delete_annotation"),
    "delete_annotation": ToolPolicy(Level.REVERSIBLE_WRITE, frozenset({"user"}), False,
                                    "Remove a note by id (undo of save_annotation)"),
    "publish_review": ToolPolicy(Level.IRREVERSIBLE_WRITE, frozenset({"user"}), True,
                                 "Write the review to the append-only published/ log (no overwrite, no delete)"),
    "execute_code_in_sandbox": ToolPolicy(Level.CODE_EXECUTION, frozenset({"critic"}), True,
                                          "Run Python in Docker (no network, 1 GB, 30 s); disabled by default"),
}


def approval_token(tool: str, payload: str, secret: str) -> str:
    """HMAC over tool+payload with a per-session secret known only to the gateway and the server."""
    return hmac.new(secret.encode(), f"{tool}:{payload}".encode(), hashlib.sha256).hexdigest()


def check_token(tool: str, payload: str, token: str) -> bool:
    """Server side: the secret arrives through the server process environment, never through tool arguments."""
    secret = os.environ.get("CRITIC_APPROVAL_SECRET", "")
    return bool(secret) and hmac.compare_digest(token, approval_token(tool, payload, secret))
