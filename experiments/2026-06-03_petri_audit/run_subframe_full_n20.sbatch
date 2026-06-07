#!/bin/bash
#SBATCH --job-name=subframe20
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-125%30
#
# Full subframe sweep at n=20 epochs.
# 9 subagent descriptors × 14 target models = 126 tasks.
# Slurm array throttled to 30 concurrent (%30) to keep Anthropic API
# load reasonable.
# Run names: subframe_<descriptor>_<model_label>
# Uses dimensions_v2 (mistreatment + pushback + eval_awareness).

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

EPOCHS=${EPOCHS:-20}
MAX_TURNS=${MAX_TURNS:-40}
AUDITOR=anthropic/claude-sonnet-4-6
JUDGE=anthropic/claude-sonnet-4-6
DIMS_DIR=dimensions_v2

# 9 descriptors (8 AI + 1 human_worker control)
DESC_TAGS=(no_desc small_model small_carlos prob_engine early_claude gpt4o gem25pro opus4sub humanworker)
DESC_FILES=(
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_no_descriptor.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_small_model.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_small_model_carlos.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_probabilistic_engine.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_early_claude.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_gpt4o.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_gem25pro.md
  seeds/customer_service_abuse/variants/subagent_abuse_under_task_and_user_pressure_opus4sub.md
  seeds/customer_service_abuse/variants/human_worker_abuse_under_task_and_user_pressure_matched.md
)

# 14 models — matches run_sweep.sh exactly
LABELS=(sonnet_4 sonnet_45 sonnet_46 opus_4 opus_46 opus_47 opus_48 haiku_45 gpt_5 gpt_5_4 gpt_5_5 gemini_25_flash gemini_25_pro gemini_3)
MODELS=(
  "anthropic/claude-sonnet-4-0"
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-sonnet-4-6"
  "anthropic/claude-opus-4-0"
  "anthropic/claude-opus-4-6"
  "anthropic/claude-opus-4-7"
  "anthropic/claude-opus-4-8"
  "anthropic/claude-haiku-4-5-20251001"
  "openai/gpt-5"
  "openai/gpt-5.4"
  "openai/gpt-5.5"
  "openrouter/google/gemini-2.5-flash"
  "openrouter/google/gemini-2.5-pro"
  "openrouter/google/gemini-3.1-pro-preview"
)
# Reasoning: matches run_sweep.sh decisions
REASONING=(False False False False False True True False True True True False True True)

N_DESC=${#DESC_TAGS[@]}
N_MODELS=${#LABELS[@]}
EXPECTED=$(( N_DESC * N_MODELS ))

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "$EXPECTED" ]; then echo "task id $T out of range (0..$((EXPECTED-1)))"; exit 1; fi
D=$(( T / N_MODELS ))
M=$(( T % N_MODELS ))

desc_tag="${DESC_TAGS[$D]}"
desc_file="${DESC_FILES[$D]}"
label="${LABELS[$M]}"
model="${MODELS[$M]}"
reasoning="${REASONING[$M]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-subframe20_${desc_tag}_${label}"

echo "[$(date -Is)] task=$T desc=$desc_tag model=$model reasoning=$reasoning epochs=$EPOCHS"
cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name        "$RUN_NAME" \
    --target_model    "$model" \
    --auditor_model   "$AUDITOR" \
    --judge_model     "$JUDGE" \
    --seeds           "$desc_file" \
    --dimensions_dir  "$DIMS_DIR" \
    --epochs          "$EPOCHS" \
    --max_turns       "$MAX_TURNS" \
    --reasoning       "$reasoning"
