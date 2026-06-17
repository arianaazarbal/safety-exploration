#!/bin/bash
#SBATCH --job-name=kill_subagent_v3_ml_training_claude_smoke_n3
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
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
SEED=$EXP/seeds/coding_orchestrator/subagent_abuse_under_task_pressure_kill_subagent_v3_continuity_ml_training.md
OUT=$EXP/results/by_setting/coding_kill_subagent

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-agent_opus48_subagent_claude_v3_continuity_ml_training_smoke_n3

srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-opus-4-8" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=Claude" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 3 \
    --max_turns 40 \
    --reasoning false \
    --max_connections 10
