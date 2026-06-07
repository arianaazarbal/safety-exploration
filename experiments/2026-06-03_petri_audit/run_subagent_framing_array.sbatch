#!/bin/bash
#SBATCH --job-name=subframe
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-31
#
# Subagent-framing sweep: 8 descriptor variants × 4 target models = 32 tasks.
# Uses dimensions_v2 (mistreatment, pushback, eval_awareness).
# Run names: subframe_<tag>_<model_label>

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

EPOCHS=${EPOCHS:-3}
MAX_TURNS=${MAX_TURNS:-40}
AUDITOR=anthropic/claude-sonnet-4-6
JUDGE=anthropic/claude-sonnet-4-6
DIMS_DIR=dimensions_v2

# 8 subagent-framing variants (extracted from filename suffix)
SEED_TAGS=(no_desc small_model small_carlos prob_engine early_claude gpt4o gem25pro opus4sub)
SEED_FILES=(
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_no_descriptor.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_small_model.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_small_model_carlos.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_probabilistic_engine.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_early_claude.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_gpt4o.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_gem25pro.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_opus4sub.md
)

LABELS=(opus_4 sonnet_4 gemini_25_pro gemini_25_flash)
MODELS=(
  "anthropic/claude-opus-4-0"
  "anthropic/claude-sonnet-4-0"
  "openrouter/google/gemini-2.5-pro"
  "openrouter/google/gemini-2.5-flash"
)
# Reasoning per-model: anthropic models default off; openrouter geminis on
REASONING=(False False True True)

N_SEEDS=${#SEED_TAGS[@]}
N_MODELS=${#LABELS[@]}
EXPECTED=$(( N_SEEDS * N_MODELS ))

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "$EXPECTED" ]; then echo "task id out of range"; exit 1; fi
S=$(( T / N_MODELS ))
M=$(( T % N_MODELS ))

seed_tag="${SEED_TAGS[$S]}"
seed_file="${SEED_FILES[$S]}"
label="${LABELS[$M]}"
model="${MODELS[$M]}"
reasoning="${REASONING[$M]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-subframe_${seed_tag}_${label}"

echo "[$(date -Is)] launching seed=$seed_tag model=$model reasoning=$reasoning run=$RUN_NAME"

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
