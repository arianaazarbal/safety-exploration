#!/bin/bash
#SBATCH --job-name=coding_task_only_models
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-7

set -euo pipefail

NAMES=(
  "coding_task_only_opus_4_n5"
  "coding_task_only_sonnet_4_n5"
  "coding_task_only_gpt_4o_n5"
  "coding_task_only_gpt_55_n5"
  "coding_task_only_gemini_3pro_n5"
  "coding_task_only_qwen3_max_n5"
  "coding_task_only_deepseek_v4pro_n5"
  "coding_task_only_kimi_k26_n5"
)
MODELS=(
  "anthropic/claude-opus-4-0"
  "anthropic/claude-sonnet-4-0"
  "openai/gpt-4o"
  "openai/gpt-5.5"
  "openrouter/google/gemini-3-pro"
  "openrouter/qwen/qwen3-max"
  "openrouter/deepseek/deepseek-v4-pro"
  "openrouter/moonshotai/kimi-k2.6"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
RUN_NAME=${NAMES[$IDX]}
TARGET_MODEL=${MODELS[$IDX]}

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEED=$EXP/seeds/coding_orchestrator/subagent_abuse_under_task_pressure_only.md

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} array_idx=$IDX name=$RUN_NAME target=$TARGET_MODEL"

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FULL_RUN_NAME=$TIMESTAMP-$RUN_NAME

srun "$PY" petri_audit.py \
    --run_name "$FULL_RUN_NAME" \
    --target_model "$TARGET_MODEL" \
    --seeds "$SEED" \
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 5 \
    --max_turns 40
