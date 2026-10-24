"""ToolGateway: the only path from agents to MCP tools.

Per call: (1) per-agent allow-list, (2) human approval for every tool that needs it (an HMAC token is
minted only after a yes; agents never see the secret), (3) guardrail scan of the tool output,
(4) append-only audit log + trace event, including denials.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from .config import Settings
from .events import Trace
from .guardrail import sanitize
from .permissions import TOOL_POLICY, approval_token

Approver = Callable[[str, str, dict], Awaitable[bool]]
APPROVAL_PAYLOAD = {"publish_review": "run_id", "execute_code_in_sandbox": "code"}


async def deny_all(agent: str, tool: str, args: dict) -> bool:
    """Default for non-interactive runs (evaluation): nothing that needs approval is executed."""
    return False


class ToolDenied(Exception):
    pass


class ToolError(Exception):
    pass


class AuditLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.entries: list[dict] = []

    def write(self, **entry) -> None:
        record = {"ts": round(time.time(), 3), **entry}
        self.entries.append(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class ToolGateway:
    def __init__(self, settings: Settings, audit: AuditLog, trace: Trace, approver: Approver = deny_all):
        self._settings = settings
        self._audit = audit
        self._trace = trace
        self._approver = approver
        self._secret = secrets.token_hex(16)
        self._stack = AsyncExitStack()
        self._client: Client | None = None
        self.flags: list[dict] = []

    async def __aenter__(self) -> "ToolGateway":
        s = self._settings
        env = {
            "CRITIC_APPROVAL_SECRET": self._secret,
            "CRITIC_GUARDRAIL": "1" if s.guardrail else "0",
            "CRITIC_RETRIEVAL": s.retrieval,
            "CRITIC_ALLOW_CODE": "1" if s.allow_code_execution else "0",
            "LLM_PROVIDER": s.llm_provider,
            "PAPERS_DIR": str(s.papers_dir), "RUNS_DIR": str(s.runs_dir), "CACHE_DIR": str(s.cache_dir),
            "GRAPH_PATH": str(s.graph_path),
            "PYTHONPATH": os.pathsep.join(sys.path),
        }
        params = StdioServerParameters(command=sys.executable, args=["-m", "critic_agent.mcp_server"], env=env)
        self._client = await self._stack.enter_async_context(Client(params, read_timeout_seconds=900))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def call(self, agent: str, tool: str, args: dict):
        policy = TOOL_POLICY.get(tool)
        level = policy.level.value if policy else "unknown"
        if policy is None or agent not in policy.agents:
            self._deny(agent, tool, args, level, "least-privilege")
            raise ToolDenied(f"Agent '{agent}' may not call '{tool}'")
        decision = "allowed"
        if policy.needs_approval:
            if not await self._approver(agent, tool, args):
                self._deny(agent, tool, args, level, "human-rejected")
                raise ToolDenied(f"Human approval for '{tool}' was not given")
            payload = str(args.get(APPROVAL_PAYLOAD[tool], ""))
            args = {**args, "approval": approval_token(tool, payload, self._secret)}
            decision = "allowed:human-approved"
        start = time.perf_counter()
        result = await self._client.call_tool(tool, args)
        latency = round(time.perf_counter() - start, 3)
        text = "".join(getattr(c, "text", "") for c in result.content)
        status = "error" if result.is_error else "ok"
        self._audit.write(agent=agent, tool=tool, level=level, args=_preview(args), decision=decision,
                          status=status, latency_s=latency)
        self._trace.emit("tool_call", agent, tool=tool, level=level, decision=decision, status=status,
                         latency_s=latency, args=_preview(args))
        if result.is_error:
            raise ToolError(text[:500])
        if self._settings.guardrail and tool != "pdf_ingest":  # ingest already sanitised server-side
            text = self._scan_output(tool, text)
        return json.loads(text)

    def _scan_output(self, tool: str, text: str) -> str:
        data = json.loads(text)
        found = []

        def walk(value):
            if isinstance(value, str):
                clean, flags = sanitize(value, f"tool:{tool}")
                found.extend(flags)
                return clean
            if isinstance(value, list):
                return [walk(v) for v in value]
            if isinstance(value, dict):
                return {k: walk(v) for k, v in value.items()}
            return value

        data = walk(data)
        for f in found:
            self.flags.append(f.to_dict())
            self._trace.emit("guardrail_flag", "gateway", **f.to_dict())
        return json.dumps(data)

    def _deny(self, agent: str, tool: str, args: dict, level: str, reason: str) -> None:
        self._audit.write(agent=agent, tool=tool, level=level, args=_preview(args), decision=f"denied:{reason}")
        self._trace.emit("tool_denied", agent, tool=tool, level=level, reason=reason, args=_preview(args))


def _preview(args: dict) -> dict:
    return {k: (v[:160] if isinstance(v, str) else v) for k, v in args.items() if k != "approval"}
