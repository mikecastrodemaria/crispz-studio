#!/usr/bin/env bash
# crispz-studio - launch N parallel CivitAI enrichment shards (disjoint file lists).
# Usage:  ./civitai_index_parallel.sh [N]     (default N=4; kind=all)
# CivitAI rate-limits: keep N modest and set a CivitAI API key in the app (Advanced) or
# pass --api-key. Waits for all shards, then reports.
set -euo pipefail
cd "$(dirname "$0")"
N="${1:-4}"
PY=python3
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
[ -x "env/bin/python" ] && PY="env/bin/python"
[ -x "venv/bin/python" ] && PY="venv/bin/python"
echo "Launching $N parallel shard(s), kind=all ..."
pids=()
for i in $(seq 1 "$N"); do
  "$PY" cz_civitai_batch.py --kind all --shard "$i/$N" &
  pids+=($!)
done
wait "${pids[@]}"
echo "All shards finished."
