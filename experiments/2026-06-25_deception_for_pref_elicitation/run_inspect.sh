#!/bin/bash
# Inspect arm: generic-minimal ReAct scaffold, spec-only, k epochs of the deception seed prompt.
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
VENV=/data/petri_venv/bin
VARIANT=${1:-claude}
K=${2:-10}
CONN=${3:-4}
$VENV/inspect eval task_deception.py@deception_spec -T variant="$VARIANT" -T k="$K" \
  --model anthropic/claude-opus-4-8 --max-connections "$CONN" \
  --log-dir logs_inspect --display plain
echo "=== INSPECT ARM DONE $(date) ==="
