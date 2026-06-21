#!/usr/bin/env bash
# Full run of the code_then_spec_blind condition (12 scenarios x k epochs), Opus generator.
# Anthropic low-prio org; ~20 concurrent connections (documented per ~/.env coordination note).
set -euo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
K="${1:-5}"
/data/petri_venv/bin/inspect eval task_blind.py@welfare_blind \
  --model anthropic/claude-opus-4-8 \
  -T k="$K" \
  --max-connections 20 \
  --log-dir logs_blind
