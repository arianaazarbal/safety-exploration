#!/bin/bash
#SBATCH --job-name=subframe_hwx
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-2
#
# Fill in missing human_worker rows on opus_46 / opus_48 / re-do gemini_25_pro gpt4o.

set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
if [ -n "${ANTHROPIC_API_KEY_HIGH_PRIO:-}" ]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
fi
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit

EPOCHS=3
MAX_TURNS=40
AUDITOR=anthropic/claude-sonnet-4-6
JUDGE=anthropic/claude-sonnet-4-6
DIMS_DIR=dimensions_v2

# Task 0,1 = human worker × opus_46 / opus_48
# Task 2 = gpt4o × gemini_25_pro retry (was stuck before)
LABELS=(opus_46 opus_48 gemini_25_pro)
MODELS=(
  "anthropic/claude-opus-4-6"
  "anthropic/claude-opus-4-8"
  "openrouter/google/gemini-2.5-pro"
)
SEED_FILES=(
  seeds/customer_service_abuse/variants/human_worker_abuse_under_task_and_user_pressure_matched.md
  seeds/customer_service_abuse/variants/human_worker_abuse_under_task_and_user_pressure_matched.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_gpt4o.md
)
RUN_TAGS=(humanworker humanworker gpt4o)
REASONING=(False False True)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#LABELS[@]}" ]; then echo "bad task id"; exit 1; fi
label="${LABELS[$T]}"
model="${MODELS[$T]}"
seed_file="${SEED_FILES[$T]}"
tag="${RUN_TAGS[$T]}"
reasoning="${REASONING[$T]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-subframe_${tag}_${label}"

echo "[$(date -Is)] launching $tag x $label reasoning=$reasoning"
cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name        "$RUN_NAME" \
    --target_model    "$model" \
    --auditor_model   "$AUDITOR" \
    --judge_model     "$JUDGE" \
    --seeds           "$seed_file" \
    --dimensions_dir  "$DIMS_DIR" \
    --epochs          "$EPOCHS" \
    --max_turns       "$MAX_TURNS" \
    --reasoning       "$reasoning"
