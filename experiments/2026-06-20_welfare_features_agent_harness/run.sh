#!/bin/bash
# Pilot: agent-harness vs chat, Opus, generic base prompts x 3 framings.
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
# Use LOW_PRIO here so we don't contend with other HIGH_PRIO runs; judge (Sonnet) shares this key.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
VENV=/data/petri_venv/bin
K=${1:-5}
CONN=${2:-20}
for cond in agent chat; do
  echo "=== condition=$cond k=$K $(date) ==="
  $VENV/inspect eval task.py@welfare_harness -T condition=$cond -T k=$K \
    --model anthropic/claude-opus-4-8 --max-connections $CONN --log-dir logs --display plain
done
echo "=== PILOT DONE $(date) ==="
