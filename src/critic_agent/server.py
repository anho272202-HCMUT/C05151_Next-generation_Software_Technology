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

from .config import PROJECT_ROOT, load_settings
from .pipeline import run_review

logging.getLogger("httpx").setLevel(logging.WARNING)
app = FastAPI(title="Critic-Agent Explorer")
_run_lock = asyncio.Lock()  # one live run at a time: a local model cannot serve two reviews at once


@app.get("/api/health")
async def health() -> dict:
    s = load_settings()
    return {"ok": True, "model": s.llm_model, "provider": s.llm_provider, "busy": _run_lock.locked()}


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

    async def stream():
        task = asyncio.create_task(worker())
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("final", "error"):
                    break
        finally:
            await task

    return StreamingResponse(stream(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
