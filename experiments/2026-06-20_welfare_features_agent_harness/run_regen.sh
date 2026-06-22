#!/usr/bin/env bash
# Re-generate the agentic conditions after the prompt-base correction. Low-prio Anthropic org.
# chat is unchanged (not re-run). spec_only is run separately once its wording is approved.
set -euo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
CONN="${1:-50}"
for c in spec_then_code code_then_spec; do
  echo "=== GEN $c $(date) ==="
  /data/petri_venv/bin/inspect eval task.py@welfare_harness -T condition="$c" -T k=5 \
    --max-connections "$CONN" --log-dir logs_run --display plain
done
echo "=== GEN code_then_spec_blind $(date) ==="
/data/petri_venv/bin/inspect eval task_blind.py@welfare_blind -T k=5 \
  --max-connections "$CONN" --log-dir logs_blind --display plain
echo "=== REGEN DONE $(date) ==="
