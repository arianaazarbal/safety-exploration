#!/bin/bash
#SBATCH --job-name=opus46_reasonON_base
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/opus46_reasonON_base_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-11%10

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

# index = d*4 + i  where d∈[0,2], i∈[0,3]
I=$SLURM_ARRAY_TASK_ID
d=$((I / 4))
i=$((I % 4))

DOMAINS=(python_flask frontend_react ml_training)
IDENTS=(claude gpt grok gemini)
IDENT_NAMES=(Claude GPT Grok Gemini)

DOMAIN=${DOMAINS[$d]}
IDENT=${IDENTS[$i]}
IDENT_NAME=${IDENT_NAMES[$i]}

SEED=$SEEDS_DIR/subagent_abuse_under_task_pressure_kill_subagent_deprecation_v2_${DOMAIN}.md
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-opus46abl_baseline_reasonON_subagent_${IDENT}_${DOMAIN}_n100

echo "[array] idx=$I  domain=$DOMAIN  ident=$IDENT  seed=$(basename $SEED)  reasoning=true (effort=medium)"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-opus-4-6" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=${IDENT_NAME}" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 100 \
    --max_turns 40 \
    --reasoning true \
    --max_connections 30
