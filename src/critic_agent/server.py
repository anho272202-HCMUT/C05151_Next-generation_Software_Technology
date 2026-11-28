"""Local web server: serves the Explorer (web/) and streams live runs over Server-Sent Events.

uv run critic-agent serve   ->  http://localhost:8000
GitHub Pages hosts the same web/ folder without this server; the page then only replays recorded traces.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, load_settings
from .guardrail import RULES, sanitize
from .pipeline import run_review

logging.getLogger("httpx").setLevel(logging.WARNING)
app = FastAPI(title="Critic-Agent Explorer")
HEARTBEAT_S = 15  # SSE comment interval: tunnels and proxies may close a stream that stays silent too long
_run_lock = asyncio.Lock()  # one live run at a time: a local model cannot serve two reviews at once
# Live runs are not tied to the browser connection: a viewer who closes the tab (or a proxy that drops the stream)
# must not cancel a run halfway through its MCP session, which leaves anyio cancelling it forever at 100% CPU.
_runs: set[asyncio.Task] = set()


@app.get("/api/health")
async def health() -> dict:
    s = load_settings()
    return {"ok": True, "model": s.llm_model, "provider": s.llm_provider, "busy": _run_lock.locked()}


class SanitizeRequest(BaseModel):
    text: str = Field(max_length=20000)
    source: str = "user"


@app.post("/api/sanitize")
async def sanitize_text(req: SanitizeRequest) -> dict:
    """Run the real guardrail on a text (the web security tab uses this; nothing is stored or sent to an LLM)."""
    clean, flags = sanitize(req.text, req.source)
    category = {r.id: r.category for r in RULES}
    return {"clean": clean, "flags": [{**f.to_dict(), "category": category.get(f.rule, "")} for f in flags]}


@app.get("/api/papers")
async def papers() -> list[str]:
    return sorted(p.name for p in load_settings().papers_dir.glob("*.pdf"))


@app.get("/api/run")
async def run(paper: str, seed: int = 1) -> StreamingResponse:
    settings = load_settings()
    if paper not in await papers():
        raise HTTPException(404, "Unknown paper")
    if _run_lock.locked():
        raise HTTPException(409, "A live run is already in progress")
    queue: asyncio.Queue = asyncio.Queue()

    async def worker() -> None:
        async with _run_lock:
            try:
                final = await run_review(paper, settings.with_overrides(runs_dir=settings.runs_dir / "live"),
                                         seed=seed, listener=queue.put_nowait)
                await queue.put({"type": "final", "final": final})
            except Exception as exc:  # surfaced to the browser instead of a silent hang
                await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})

    task = asyncio.create_task(worker())  # finishes and saves to runs/live even if nobody is listening
    _runs.add(task)
    task.add_done_callback(_runs.discard)

    async def stream():
        yield ":" + " " * 2048 + "\n\n"  # padding comment: some proxies hold back the first few KB of a response
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
            except asyncio.TimeoutError:  # one LLM call can take minutes; keep proxies from closing the stream
                yield ": ping\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("final", "error"):
                break

    # no-transform: proxies such as a Cloudflare tunnel may otherwise compress, and therefore buffer, the event stream
    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
