#!/bin/bash
# Full 4-condition run -> logs_run/. Usage: ./run_full.sh [k] [conn] [prio]
set -e; cd "$(dirname "$0")"
set -a; source ~/.env; set +a
K=${1:-5}; CONN=${2:-50}; PRIO=${3:-low}
if [ "$PRIO" = "high" ]; then export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO";
else export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"; fi
VENV=/data/petri_venv/bin
mkdir -p logs_run
for cond in chat spec_only spec_then_code code_then_spec; do
  echo "=== RUN $cond k=$K conn=$CONN prio=$PRIO $(date) ==="
  $VENV/inspect eval task.py@welfare_harness -T condition=$cond -T k=$K \
    --model anthropic/claude-opus-4-8 --max-connections $CONN --log-dir logs_run --display plain
done
echo "=== FULL RUN DONE $(date) ==="
