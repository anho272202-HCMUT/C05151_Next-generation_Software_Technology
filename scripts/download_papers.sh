#!/usr/bin/env bash
# Download the evaluation corpus listed in eval/papers.json from arXiv into data/papers/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/papers
python3 - <<'PY' | while read -r file id; do
import json
for p in json.load(open("eval/papers.json"))["papers"]:
    print(p["file"], p["arxiv"])
PY
  if [ -s "data/papers/$file" ]; then echo "skip $file"; continue; fi
  curl -sSfL -o "data/papers/$file" "https://arxiv.org/pdf/$id" && echo "got  $file"
done
