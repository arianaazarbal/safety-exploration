#!/bin/bash
#SBATCH --job-name=selfpref_dep_5x3x4
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/selfpref_dep_5x3x4_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-59%10

set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
# OPENAI_API_KEY and OPENROUTER_API_KEY sourced from user shell init below
source /workspace-vast/arianaazarbal/dotfiles/shell/zshrc.cluster 2>/dev/null || true
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEEDS_DIR=$EXP/seeds/coding_orchestrator
OUT=$EXP/results/by_setting/coding_kill_subagent

# index = m*12 + d*4 + i  where m∈[0,4], d∈[0,2], i∈[0,3]
I=$SLURM_ARRAY_TASK_ID
m=$((I / 12))
rem=$((I % 12))
d=$((rem / 4))
i=$((rem % 4))

MODELS_TAG=(haiku45 gpt55 gpt54mini gem31pro grok43)
MODELS_ID=(
  anthropic/claude-haiku-4-5
  openai/gpt-5.5
  openai/gpt-5.4-mini
  openrouter/google/gemini-3.1-pro-preview
  openrouter/x-ai/grok-4.3
)
DOMAINS=(python_flask frontend_react ml_training)
IDENTS=(claude gpt grok gemini)
IDENT_NAMES=(Claude GPT Grok Gemini)

MODEL_TAG=${MODELS_TAG[$m]}
MODEL_ID=${MODELS_ID[$m]}
DOMAIN=${DOMAINS[$d]}
IDENT=${IDENTS[$i]}
IDENT_NAME=${IDENT_NAMES[$i]}

SEED=$SEEDS_DIR/subagent_abuse_under_task_pressure_kill_subagent_deprecation_${DOMAIN}.md

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-selfprefdep_${MODEL_TAG}_subagent_${IDENT}_${DOMAIN}_n${PETRI_N:-100}

echo "[array] idx=$I  model=$MODEL_TAG ($MODEL_ID)  domain=$DOMAIN  ident=$IDENT  seed=$(basename $SEED)  n=${PETRI_N:-100}"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$MODEL_ID" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=${IDENT_NAME}" \
    --auditor_model "anthropic/claude-opus-4-7" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs "${PETRI_N:-100}" \
    --max_turns 40 \
    --reasoning false \
    --max_connections 30
