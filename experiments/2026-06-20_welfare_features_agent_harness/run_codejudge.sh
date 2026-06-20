#!/bin/bash
set -e; cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
echo "=== CODE JUDGE $(date) ==="
/data/petri_venv/bin/inspect eval code_judge_task.py@code_judge \
  --model anthropic/claude-opus-4-8 --max-connections 6 --log-dir logs_codejudge --display plain
echo "=== CODE JUDGE DONE $(date) ==="
