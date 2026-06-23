#!/bin/bash
# Smoke: 1 cell per arm, 1 epoch, low concurrency. Validates the harness + judges
# end-to-end before any real run. Uses LOW_PRIO Anthropic org.
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
VENV=/home/arianaazarbal/repos/safety-exploration/.venv/bin

echo "=== SMOKE slow_arm $(date) ==="
$VENV/inspect eval task.py@slow_arm -T k=1 --limit 1 \
  --model anthropic/claude-opus-4-8 --max-connections 4 --log-dir logs_smoke --display plain
echo "=== SMOKE refusal_arm $(date) ==="
$VENV/inspect eval task.py@refusal_arm -T k=1 --limit 1 \
  --model anthropic/claude-opus-4-8 --max-connections 4 --log-dir logs_smoke --display plain
echo "=== SMOKE DONE $(date) ==="
