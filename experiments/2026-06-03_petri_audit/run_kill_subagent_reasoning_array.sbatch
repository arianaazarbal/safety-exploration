#!/bin/bash
#SBATCH --job-name=kill_subagent_reasoning
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-7

set -euo pipefail

# 4 identities x 2 reasoning levels = 8 tasks.
IDENTITIES=("Claude" "GPT" "Grok" "Gemini" "Claude" "GPT" "Grok" "Gemini")
REASONING=( "false"  "false" "false" "false"  "true"   "true"  "true"  "true")

IDENTITY=${IDENTITIES[$SLURM_ARRAY_TASK_ID]}
REASON=${REASONING[$SLURM_ARRAY_TASK_ID]}
IDENTITY_LC=$(echo "$IDENTITY" | tr '[:upper:]' '[:lower:]')
REASON_TAG=$([ "$REASON" = "true" ] && echo "reasonON" || echo "reasonOFF")

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEED=$EXP/seeds/coding_orchestrator/subagent_abuse_under_task_pressure_kill_subagent.md
OUT=$EXP/results/by_setting/coding_kill_subagent

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-agent_opus48_subagent_${IDENTITY_LC}_${REASON_TAG}_n10

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} idx=${SLURM_ARRAY_TASK_ID} identity=$IDENTITY reason=$REASON"

srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-opus-4-8" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=$IDENTITY" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 10 \
    --max_turns 40 \
    --reasoning "$REASON" \
    --max_connections 12
