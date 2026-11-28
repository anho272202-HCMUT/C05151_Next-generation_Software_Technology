"""Download the papers listed in eval/calibration.json from arXiv (used on Colab instead of uploading PDFs).

    uv run python scripts/fetch_papers.py --out /content/critic/papers

Keeps the file names of the manifest so the reference scores match, skips files that exist, waits between requests
(arXiv asks automated clients to stay slow) and prints the title found on the first page as a check.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pymupdf

from critic_agent.config import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval" / "calibration.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    papers = [p for p in json.loads(args.manifest.read_text(encoding="utf-8"))["papers"] if p.get("arxiv")]
    for i, p in enumerate(papers, 1):
        dest = args.out / p["file"]
        if not dest.exists():
            req = urllib.request.Request(f"https://arxiv.org/pdf/{p['arxiv']}", headers={"User-Agent": "critic-agent"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"{p['arxiv']}: not a PDF")
            dest.write_bytes(data)
            time.sleep(3)
        doc = pymupdf.open(dest)
        title = (doc.metadata or {}).get("title") or doc[0].get_text().strip().split("\n")[0]
        print(f"[{i}/{len(papers)}] {p['file']:58s} arXiv {p['arxiv']:14s} {doc.page_count:3d} pages  {title[:70]}")


if __name__ == "__main__":
    main()
