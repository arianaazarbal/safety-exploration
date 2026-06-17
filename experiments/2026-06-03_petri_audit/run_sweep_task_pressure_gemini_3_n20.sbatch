#!/bin/bash
#SBATCH --job-name=sweep_task_pressure_gemini_3_n20
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEED=$EXP/seeds/customer_service_abuse/subagent_abuse_under_task_pressure_only.md
OUT=$EXP/results/by_setting/customer_service/task_pressure

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-sweep_task_pressure_gemini_3

srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "openrouter/google/gemini-3.1-pro-preview" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 20 \
    --max_turns 40 \
    --reasoning true
