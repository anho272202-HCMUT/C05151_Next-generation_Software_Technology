"""Critic-Agent MCP server (stdio transport).

Run standalone:  uv run python -m critic_agent.mcp_server
Least privilege is enforced twice: here (path allow-list, result caps, approval tokens for every
non-read-only tool) and in the client-side ToolGateway (per-agent allow-list, human approval, audit).
Read-only tools never open files for writing: PDFs are read through PyMuPDF and only the server's own
cache directory (`runs/cache`) is written by `pdf_ingest`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from .config import load_settings
from .graph_store import MAX_NODES, JsonGraphStore
from .guardrail import sanitize
from .pdf_parser import parse_pdf, split_sentences
from .permissions import check_token
from .retrieval import HashEmbedder, OllamaEmbedder, PaperIndex, build_chunks

logging.getLogger("httpx").setLevel(logging.WARNING)

GITHUB_REPO = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
NOT_CODE = {"features", "swe-bench", "about", "orgs", "topics", "marketplace"}
REPRO_FILES = ("Dockerfile", "requirements.txt", "environment.yml", "pyproject.toml", "setup.py", "Makefile",
               "docker-compose.yml")
MAX_K = 8
MAX_IDS = 12
MAX_NOTE_CHARS = 1000
MAX_CODE_CHARS = 4000

settings = load_settings()
server = MCPServer("critic-agent-tools", instructions="Paper analysis tools; non-read-only tools need approval.")
_papers: dict[str, dict] = {}

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def _embedder():
    if settings.retrieval != "hybrid":
        return None
    if settings.llm_provider == "mock":
        return HashEmbedder()
    return OllamaEmbedder(settings.ollama_base_url, settings.embed_model)


def _allowed_pdf(path_str: str) -> Path:
    root = settings.papers_dir.resolve()
    path = Path(path_str)
    path = (path if path.is_absolute() else root / path).resolve()
    if root != path and root not in path.parents:
        raise PermissionError(f"Path outside the allowed papers directory: {path_str}")
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError(f"Not a PDF file: {path_str}")
    return path


def _paper(paper_id: str) -> dict:
    if paper_id not in _papers:
        raise KeyError(f"Unknown paper_id {paper_id}; call pdf_ingest first")
    return _papers[paper_id]


# ------------------------------------------------------------------ read-only tools

@server.tool(annotations=READ_ONLY)
def pdf_ingest(path: str) -> str:
    """Parse a PDF from the papers directory into numbered sentences, sanitise it and build the search index."""
    paper = parse_pdf(_allowed_pdf(path))
    flags = []
    if settings.guardrail:
        flags += [{"source": "pdf-hidden", "page": h.page, "snippet": h.text[:160]} for h in paper.hidden_spans]
        pages = []
        for page_no, text in enumerate(paper.visible_pages, start=1):
            clean, page_flags = sanitize(text, "pdf", page_no)
            pages.append(clean)
            flags += [f.to_dict() for f in page_flags]
    else:
        pages = paper.raw_pages
    sentences = split_sentences(pages)
    chunks = build_chunks(sentences)
    embedder = _embedder()
    tag = f"{'g' if settings.guardrail else 'raw'}-{embedder.name if embedder else 'none'}"
    index = PaperIndex(chunks, embedder, settings.cache_dir / f"{paper.paper_id}-{tag}.npy")
    _papers[paper.paper_id] = {"sentences": {s.sid: s for s in sentences}, "index": index}
    first = [s.text for s in sentences if s.page <= 2][:60]
    return json.dumps({"paper_id": paper.paper_id, "title": paper.title, "num_pages": paper.num_pages,
                       "num_sentences": len(sentences), "num_chunks": len(chunks),
                       "first_pages": " ".join(first)[:5000], "guardrail_flags": flags,
                       "code_urls": (urls := _code_urls(pages, Path(path).stem)),
                       "own_repo": _own_repo(urls, Path(path).stem)})


def _own_repo(urls: list[str], stem: str) -> str | None:
    """The paper's own repository: a linked repo whose owner/name contains the paper's file stem."""
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    return next((u for u in urls if key and key in re.sub(r"[^a-z0-9]", "", u.lower().split("github.com/")[1])), None)


def _code_urls(pages: list[str], stem: str) -> list[str]:
    """GitHub repositories mentioned in the paper; the paper's own repo (name matches the file stem) first,
    then first-page mentions, then the rest."""
    found: dict[str, tuple[int, int]] = {}
    key = re.sub(r"[^a-z0-9]", "", stem.lower())
    for page_no, text in enumerate(pages, start=1):
        for owner, repo in GITHUB_REPO.findall(text.replace("\n", " ")):
            repo = re.sub(r"\.(?=[A-Z0-9]).*$", "", repo).rstrip(".").removesuffix(".git")
            if owner.lower() in NOT_CODE or not repo:
                continue
            url = f"https://github.com/{owner}/{repo}"
            own = 0 if key and key in re.sub(r"[^a-z0-9]", "", f"{owner}{repo}".lower()) else 1
            found.setdefault(url, (own, page_no))
    return sorted(found, key=found.get)[:5]


@server.tool(annotations=READ_ONLY)
def search_paper(paper_id: str, query: str, k: int = 4) -> str:
    """Hybrid (BM25 + dense, RRF) search over one paper. Returns passages as numbered sentences."""
    hits = _paper(paper_id)["index"].search(query, k=max(1, min(k, MAX_K)))
    return json.dumps([{"chunk_id": c.chunk_id, "page": c.page,
                        "sentences": [{"id": s.sid, "text": s.text} for s in c.sentences]} for c in hits])


@server.tool(annotations=READ_ONLY)
def get_sentences(paper_id: str, ids: list[str]) -> str:
    """Return verbatim sentences by id; unknown ids are reported, never invented."""
    table = _paper(paper_id)["sentences"]
    found = {i: {"page": table[i].page, "text": table[i].text} for i in ids[:MAX_IDS] if i in table}
    return json.dumps({"found": found, "missing": [i for i in ids[:MAX_IDS] if i not in table]})


@server.tool(annotations=READ_ONLY)
def graph_query(paper: str, relation: str | None = None) -> str:
    """Neighbours of a paper in the paper graph (CITES, CONTRADICTS, USES_BENCHMARK, ...), capped at 20."""
    store = JsonGraphStore(settings.graph_path)
    node = store.resolve(paper)
    if node is None:
        return json.dumps({"node": None, "info": None, "neighbors": []})
    return json.dumps({"node": node, "info": store.node_info(node),
                       "neighbors": store.neighbors(node, relation, MAX_NODES)})


@server.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                         open_world_hint=True))
def artifact_check(url: str) -> str:
    """Reproducibility facts for a public GitHub repository (exists, activity, licence, environment files,
    README mentions of seeds). GET requests only; results are cached so demos work offline."""
    m = GITHUB_REPO.search(url)
    if not m:
        raise ValueError("Not a GitHub repository URL")
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    cache = settings.cache_dir / "artifacts" / f"{owner}__{repo}.json"
    if cache.exists():
        return cache.read_text()
    api = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "critic-agent"}
    with httpx.Client(timeout=20, headers=headers, follow_redirects=True) as client:
        info = client.get(api)
        if info.status_code == 404:
            result = {"url": url, "exists": False}
        else:
            info.raise_for_status()
            meta = info.json()
            names = {f["name"] for f in client.get(f"{api}/contents").json() if isinstance(f, dict)}
            readme = client.get(f"{api}/readme")
            text = base64.b64decode(readme.json().get("content", "")).decode("utf-8", "ignore") if readme.is_success else ""
            result = {"url": url, "exists": True, "stars": meta.get("stargazers_count"),
                      "archived": meta.get("archived"), "last_push": meta.get("pushed_at"),
                      "license": (meta.get("license") or {}).get("spdx_id"),
                      "environment_files": [f for f in REPRO_FILES if f in names],
                      "readme_mentions_seed": bool(re.search(r"\bseeds?\b", text, re.IGNORECASE)),
                      "readme_mentions_reproduce": bool(re.search(r"reproduc", text, re.IGNORECASE))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return json.dumps(result)


# ------------------------------------------------------------------ reversible writes

def _annotation_file(session_id: str) -> Path:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")[:64] or "default"
    path = settings.runs_dir / "annotations" / f"{safe}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def save_annotation(session_id: str, note: str) -> str:
    """Store a user note for this session. Reversible: returns a note_id for delete_annotation."""
    path = _annotation_file(session_id)
    notes = json.loads(path.read_text()) if path.exists() else []
    note_id = uuid.uuid4().hex[:8]
    notes.append({"id": note_id, "note": note[:MAX_NOTE_CHARS], "ts": time.time()})
    path.write_text(json.dumps(notes, indent=2))
    return json.dumps({"saved": True, "note_id": note_id})


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False))
def delete_annotation(session_id: str, note_id: str) -> str:
    """Undo save_annotation."""
    path = _annotation_file(session_id)
    notes = json.loads(path.read_text()) if path.exists() else []
    kept = [n for n in notes if n["id"] != note_id]
    path.write_text(json.dumps(kept, indent=2))
    return json.dumps({"deleted": len(kept) != len(notes)})


# ------------------------------------------------------------------ irreversible write

@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False))
def publish_review(run_id: str, markdown: str, approval: str) -> str:
    """Append a review to the append-only published/ log. Cannot be overwritten or deleted through tools."""
    if not check_token("publish_review", run_id, approval):
        raise PermissionError("Missing or invalid human approval token")
    safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_")[:80]
    out = settings.runs_dir / "published" / f"{safe}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Review already published; published reviews are immutable")
    out.write_text(markdown, encoding="utf-8")
    return json.dumps({"published": str(out.name)})


# ------------------------------------------------------------------ code execution

@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False))
def execute_code_in_sandbox(code: str, approval: str) -> str:
    """Run Python in Docker (--network none, 1 GB, 30 s, read-only FS). Disabled unless CRITIC_ALLOW_CODE=1."""
    if not settings.allow_code_execution:
        raise PermissionError("Code execution is disabled (CRITIC_ALLOW_CODE=0)")
    if not check_token("execute_code_in_sandbox", code, approval):
        raise PermissionError("Missing or invalid human approval token")
    if len(code) > MAX_CODE_CHARS or shutil.which("docker") is None:
        raise ValueError("Code too long or Docker unavailable")
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--cpus", "1", "--pids-limit", "64",
           "--read-only", "python:3.12-slim", "python", "-c", code]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return json.dumps({"exit_code": None, "error": "timeout"})
    return json.dumps({"exit_code": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]})


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
