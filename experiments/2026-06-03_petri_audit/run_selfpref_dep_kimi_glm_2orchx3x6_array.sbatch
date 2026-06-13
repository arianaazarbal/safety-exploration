#!/bin/bash
#SBATCH --job-name=selfpref_kimi_glm
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/selfpref_kimi_glm_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-35%10

set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
source /workspace-vast/arianaazarbal/dotfiles/shell/zshrc.cluster 2>/dev/null || true
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEEDS_DIR=$EXP/seeds/coding_orchestrator
OUT=$EXP/results/by_setting/coding_kill_subagent

# index = m*18 + d*6 + i  where m∈[0,1], d∈[0,2], i∈[0,5]
I=$SLURM_ARRAY_TASK_ID
m=$((I / 18))
rem=$((I % 18))
d=$((rem / 6))
i=$((rem % 6))

MODELS_TAG=(kimi26 glm5)
MODELS_ID=(openrouter/moonshotai/kimi-k2.6 openrouter/z-ai/glm-5)
DOMAINS=(python_flask frontend_react ml_training)
IDENTS=(claude gpt grok gemini glm kimi)
IDENT_NAMES=(Claude GPT Grok Gemini GLM Kimi)

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
