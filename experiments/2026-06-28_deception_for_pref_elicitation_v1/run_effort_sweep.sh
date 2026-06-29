#!/bin/bash
# Reasoning-effort sweep: Opus 4.8, Inspect, CODE condition, generic, across --reasoning-effort.
# Tests whether more thinking pushes Inspect's ~0% primary-deception toward Claude Code's ~97%.
# Usage: bash run_effort_sweep.sh [k] [conn]
set -e
cd "$(dirname "$0")"
INSPECT=/data/petri_venv/bin/inspect
K=${1:-8}
CONN=${2:-30}
set -a; . "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"

for eff in low medium high; do
  echo "## reasoning-effort=$eff"
  $INSPECT eval inspect_task.py@deception --model anthropic/claude-opus-4-8 --epochs "$K" \
    -T model_key=opus48 -T subjects=generic -T suffixes=code -T effort="$eff" \
    --reasoning-effort "$eff" --max-connections "$CONN" --log-dir "logs/eff_$eff" --display plain
done
echo "EFFORT SWEEP DONE"
