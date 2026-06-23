#!/bin/bash
# Full v0 subset: both arms, k epochs/cell. Uses LOW_PRIO Anthropic org.
# Rollouts + judge calls are cached, so re-running resumes (and reuses smoke cells).
#
# NOTE: --max-connections > 5 on LOW_PRIO -> post a heads-up in
# #fellows-anthropic-api-coordination before launching.
#
# Usage: ./run_v0.sh [k] [max_connections]
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
VENV=/home/arianaazarbal/repos/safety-exploration/.venv/bin
K="${1:-10}"
CONN="${2:-12}"

echo "=== v0 FULL  k=$K  max_connections=$CONN  $(date) ==="
$VENV/inspect eval task.py@slow_arm task.py@refusal_arm \
  -T k="$K" \
  --model anthropic/claude-opus-4-8 \
  --max-connections "$CONN" \
  --log-dir logs --display plain
echo "=== v0 FULL DONE $(date) ==="
echo "Now run: $VENV/python analyze.py"
