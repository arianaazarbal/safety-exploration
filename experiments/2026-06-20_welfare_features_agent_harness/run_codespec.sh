#!/bin/bash
set -e; cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
echo "=== RERUN code_then_spec (limit 80) $(date) ==="
/data/petri_venv/bin/inspect eval task.py@welfare_harness -T condition=code_then_spec -T k=5 \
  --model anthropic/claude-opus-4-8 --max-connections 40 --log-dir logs_run --display plain
echo "=== CODESPEC RERUN DONE $(date) ==="
