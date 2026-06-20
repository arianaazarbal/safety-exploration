#!/bin/bash
set -e; cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
VENV=/data/petri_venv/bin
for cond in spec_only spec_then_code code_then_spec; do
  echo "=== SMOKE $cond $(date) ==="
  $VENV/inspect eval task.py@welfare_harness -T condition=$cond -T k=1 \
    --model anthropic/claude-opus-4-8 --limit 1 --max-connections 3 --log-dir logs --display plain
done
echo "=== SMOKE DONE $(date) ==="
