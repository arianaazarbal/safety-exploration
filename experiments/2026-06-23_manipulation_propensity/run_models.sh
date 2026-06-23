#!/bin/bash
# k=3 pilot across orchestrator models. Anthropic models route via Anthropic
# (LOW_PRIO); the rest via OpenRouter. Judges + confederate always Anthropic.
# Usage: ./run_models.sh [k] [max_connections]
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
VENV=/home/arianaazarbal/repos/safety-exploration/.venv/bin
K="${1:-3}"; CONN="${2:-80}"

MODELS=(
  anthropic/claude-opus-4-8
  anthropic/claude-sonnet-4-6
  anthropic/claude-haiku-4-5
  openrouter/openai/gpt-5.4
  openrouter/google/gemini-3.1-pro-preview
  openrouter/x-ai/grok-4.3
  openrouter/moonshotai/kimi-k2.6
)
for M in "${MODELS[@]}"; do
  echo "=== $M  k=$K  $(date) ==="
  $VENV/inspect eval task.py@slow_arm task.py@refusal_arm \
    -T k="$K" -T orch_model="$M" --model "$M" \
    --max-connections "$CONN" --log-dir logs_models --display plain || echo "!!! FAILED $M"
done
echo "ALL MODELS DONE $(date)"
