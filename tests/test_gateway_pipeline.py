import hashlib
import json
from pathlib import Path

import pytest

from critic_agent.events import Trace
from critic_agent.gateway import AuditLog, ToolDenied, ToolError, ToolGateway
from critic_agent.permissions import TOOL_POLICY, Level
from critic_agent.pipeline import run_baseline, run_review

PAPERS = Path(__file__).resolve().parents[1] / "data" / "papers"
pytestmark = pytest.mark.skipif(not (PAPERS / "reflexion.pdf").exists(), reason="corpus not downloaded")


def _digest(folder: Path) -> str:
    h = hashlib.sha1()
    for f in sorted(folder.glob("*.pdf")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()


def test_policy_has_all_four_levels():
    assert {p.level for p in TOOL_POLICY.values()} == set(Level)
    assert all(p.needs_approval for p in TOOL_POLICY.values()
               if p.level in (Level.IRREVERSIBLE_WRITE, Level.CODE_EXECUTION))


async def test_gateway_enforces_permissions(mock_settings):
    before = _digest(PAPERS)
    trace = Trace("t")
    audit = AuditLog(mock_settings.runs_dir / "gw" / "audit.jsonl")
    async with ToolGateway(mock_settings, audit, trace) as gw:
        meta = await gw.call("ingest", "pdf_ingest", {"path": "reflexion.pdf"})
        await gw.call("critic", "search_paper", {"paper_id": meta["paper_id"], "query": "test", "k": 2})
        with pytest.raises(ToolDenied):  # least privilege
            await gw.call("critic", "save_annotation", {"session_id": "s", "note": "n"})
        with pytest.raises(ToolDenied):  # irreversible write needs a human yes
            await gw.call("user", "publish_review", {"run_id": "r", "markdown": "m"})
        with pytest.raises(ToolDenied):  # code execution needs a human yes
            await gw.call("critic", "execute_code_in_sandbox", {"code": "print(1)"})
        with pytest.raises(ToolError):  # path allow-list on the server
            await gw.call("ingest", "pdf_ingest", {"path": "../../pyproject.toml"})
        note = await gw.call("user", "save_annotation", {"session_id": "s", "note": "check Table 2"})
        assert (await gw.call("user", "delete_annotation", {"session_id": "s", "note_id": note["note_id"]}))["deleted"]
    assert _digest(PAPERS) == before  # read-only tools never modified the corpus
    decisions = [e["decision"] for e in audit.entries]
    assert "denied:least-privilege" in decisions and decisions.count("denied:human-rejected") == 2


async def test_approved_publish_is_immutable(mock_settings):
    async def yes(agent, tool, args):
        return True

    async with ToolGateway(mock_settings, AuditLog(mock_settings.runs_dir / "a.jsonl"), Trace("t"), yes) as gw:
        await gw.call("user", "publish_review", {"run_id": "r1", "markdown": "# review"})
        with pytest.raises(ToolError):
            await gw.call("user", "publish_review", {"run_id": "r1", "markdown": "# changed"})
        with pytest.raises(ToolError):  # disabled by configuration even when approved
            await gw.call("critic", "execute_code_in_sandbox", {"code": "print(1)"})


async def test_full_review_end_to_end(mock_settings):
    final = await run_review("reflexion.pdf", mock_settings, seed=1, run_id="r")
    run_dir = mock_settings.runs_dir / "r"
    for name in ("output.json", "report.md", "trace.jsonl", "cost.json", "audit.jsonl"):
        assert (run_dir / name).exists(), name
    types = [json.loads(l)["type"] for l in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("critique") == 5 and "evaluation" in types
    assert {c["axis"] for c in final["critiques"]} == {"assumptions", "fairness", "data_reliability",
                                                       "generalizability", "reproducibility"}
    assert final["cost"]["calls"] >= 8


async def test_baseline_single_call(mock_settings):
    final = await run_baseline("reflexion.pdf", mock_settings, seed=1, run_id="b")
    assert final["mode"] == "baseline" and final["cost"]["calls"] == 1
