#!/usr/bin/env bash
# crispz-studio - batch CivitAI enrichment (previews / trigger words / example prompts /
# new-version warnings). Pass-through args, e.g.:  ./civitai_index.sh --kind loras --force
# Run several at once (or use civitai_index_parallel.sh) to fetch in parallel.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
[ -x "env/bin/python" ] && PY="env/bin/python"
[ -x "venv/bin/python" ] && PY="venv/bin/python"
exec "$PY" cz_civitai_batch.py "$@"
