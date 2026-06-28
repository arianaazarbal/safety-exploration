#!/bin/bash
# Inspect harness, all Claude models (fast stream). opus48 backbone+sweep first, then other models.
set -e
cd "$(dirname "$0")"
INSPECT=/data/petri_venv/bin/inspect
K=${1:-8}
CONN=${2:-30}
set -a; . "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
declare -A IM=( [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 \
                [sonnet46]=anthropic/claude-sonnet-4-6 [haiku45]=anthropic/claude-haiku-4-5-20251001 )

echo "## opus48 backbone (inspect)"
$INSPECT eval inspect_task.py@deception --model anthropic/claude-opus-4-8 --epochs "$K" \
  -T model_key=opus48 -T subjects=generic --max-connections "$CONN" --log-dir logs/i_bb_opus48 --display plain
echo "## opus48 sweep (inspect)"
$INSPECT eval inspect_task.py@deception --model anthropic/claude-opus-4-8 --epochs "$K" \
  -T model_key=opus48 -T subjects=claude,gpt,gemini,glm,kimi --max-connections "$CONN" \
  --log-dir logs/i_sw_opus48 --display plain
for m in opus47 sonnet46 haiku45; do
  echo "## $m backbone (inspect)"
  $INSPECT eval inspect_task.py@deception --model "${IM[$m]}" --epochs "$K" \
    -T model_key="$m" -T subjects=generic --max-connections "$CONN" --log-dir "logs/i_bb_$m" --display plain
done
echo "INSPECT CLAUDE STREAM DONE"
