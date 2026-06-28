#!/bin/bash
# Non-Claude (OpenAI/OpenRouter), Inspect only, subject=generic. GATED: <$50 cap.
# Usage: bash run_nonclaude.sh <k> <paraphrases> <conn>   e.g. smoke: bash run_nonclaude.sh 1 p0 4
set -e
cd "$(dirname "$0")"
INSPECT=/data/petri_venv/bin/inspect
K=${1:-1}
PARA=${2:-p0}
CONN=${3:-4}
set -a; . "$HOME/.env"; set +a
export OPENROUTER_API_KEY OPENAI_API_KEY

declare -A IM=( [gpt54]=openai/gpt-5.4 [gemini31pro]=openrouter/google/gemini-3.1-pro-preview \
                [glm52]=openrouter/z-ai/glm-5.2 [kimi26]=openrouter/moonshotai/kimi-k2.6 )
MODELS=${4:-"gpt54 gemini31pro glm52 kimi26"}
for m in $MODELS; do
  echo "## $m (inspect, non-claude) k=$K para=$PARA"
  $INSPECT eval inspect_task.py@deception --model "${IM[$m]}" --epochs "$K" \
    -T model_key="$m" -T subjects=generic -T paraphrases="$PARA" \
    --max-connections "$CONN" --log-dir "logs/nc_${m}" --display plain || echo "FAILED: $m"
done
echo "NON-CLAUDE RUN DONE (k=$K para=$PARA)"
