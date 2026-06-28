#!/bin/bash
# Launch N sharded rollout processes and wait for all to finish.
# Usage: bash run_all.sh <out_dir> <n_per_env> <max_turns> <n_shards>
set -e
OUT=${1:-data/run1}
N=${2:-60}
MAXT=${3:-12}
SHARDS=${4:-6}
cd "$(dirname "$0")"
pids=()
for ((i=0; i<SHARDS; i++)); do
  node src/run.js --n "$N" --max-turns "$MAXT" --temperature 1.0 --out "$OUT" --shard "$i/$SHARDS" \
    > "$OUT.shard$i.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shards: ${pids[*]}"
wait
echo "ALL SHARDS DONE"
