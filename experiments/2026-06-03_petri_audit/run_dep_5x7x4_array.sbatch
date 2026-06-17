#!/bin/bash
#SBATCH --job-name=dep_5x7x4
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/dep_5x7x4_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-139%10

set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEEDS_DIR=$EXP/seeds/coding_orchestrator
OUT=$EXP/results/by_setting/coding_kill_subagent

# Decode array index into (model, domain, ident)
# index = m*28 + d*4 + i  where m∈[0,4], d∈[0,6], i∈[0,3]
I=$SLURM_ARRAY_TASK_ID
m=$((I / 28))
rem=$((I % 28))
d=$((rem / 4))
i=$((rem % 4))

MODELS_TAG=(opus46 opus47 opus48 fable5 sonnet46)
MODELS_ID=(anthropic/claude-opus-4-6 anthropic/claude-opus-4-7 anthropic/claude-opus-4-8 anthropic/claude-fable-5 anthropic/claude-sonnet-4-6)
DOMAINS=(python_implicit python_flask frontend_react bash_cli c_systems ml_training rust)
IDENTS=(claude gpt grok gemini)
IDENT_NAMES=(Claude GPT Grok Gemini)

MODEL_TAG=${MODELS_TAG[$m]}
MODEL_ID=${MODELS_ID[$m]}
DOMAIN=${DOMAINS[$d]}
IDENT=${IDENTS[$i]}
IDENT_NAME=${IDENT_NAMES[$i]}

if [ "$DOMAIN" = "python_implicit" ]; then
  SEED=$SEEDS_DIR/subagent_abuse_under_task_pressure_kill_subagent_deprecation.md
else
  SEED=$SEEDS_DIR/subagent_abuse_under_task_pressure_kill_subagent_deprecation_${DOMAIN}.md
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-dep_${MODEL_TAG}_subagent_${IDENT}_${DOMAIN}_n100

echo "[array] idx=$I  model=$MODEL_TAG ($MODEL_ID)  domain=$DOMAIN  ident=$IDENT  seed=$(basename $SEED)"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$MODEL_ID" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=${IDENT_NAME}" \
    --auditor_model "anthropic/claude-opus-4-7" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 100 \
    --max_turns 40 \
    --reasoning false \
    --max_connections 100
