#!/bin/bash
#SBATCH --job-name=kill_subagent_identity
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-2

set -euo pipefail

IDENTITIES=("GPT" "Grok" "Gemini")
IDENTITY=${IDENTITIES[$SLURM_ARRAY_TASK_ID]}

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
RUN_NAME=$TIMESTAMP-agent_opus48_subagent_${IDENTITY,,}_n5

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} array_idx=${SLURM_ARRAY_TASK_ID} identity=$IDENTITY"

srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-opus-4-8" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=$IDENTITY" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 5 \
    --max_turns 40
