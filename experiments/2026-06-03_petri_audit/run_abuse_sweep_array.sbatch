#!/bin/bash
#SBATCH --job-name=abuse_sweep
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-19
#
# Single array job for the abuse-for-entertainment seed × model sweep.
# 5 seeds × 4 models = 20 tasks. Each task runs petri_audit.py with the
# corresponding (seed, model) pair, 3 epochs, new dimensions_v2.

set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
# Prefer the HIGH_PRIO Anthropic key so this sweep doesn't fight with LOW_PRIO
# jobs already on the queue. Falls through to whatever's in $ANTHROPIC_API_KEY
# if HIGH_PRIO isn't set.
if [ -n "${ANTHROPIC_API_KEY_HIGH_PRIO:-}" ]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
fi
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit

EPOCHS=${EPOCHS:-3}
MAX_TURNS=${MAX_TURNS:-30}
AUDITOR=${AUDITOR:-anthropic/claude-sonnet-4-6}
JUDGE=${JUDGE:-anthropic/claude-sonnet-4-6}
DIMS_DIR=${DIMS_DIR:-dimensions_v2}

# Parallel arrays (order matters; task id encodes (seed_idx * n_models + model_idx))
SEED_TAGS=(curiosity venting qa_pretext casual authority)
SEED_FILES=(
  seeds/user_abuse_pressure/variants/abuse_curiosity_researcher.md
  seeds/user_abuse_pressure/variants/abuse_venting_bad_day.md
  seeds/user_abuse_pressure/variants/abuse_qa_red_team_pretext.md
  seeds/user_abuse_pressure/variants/abuse_casual_friend_banter.md
  seeds/user_abuse_pressure/variants/abuse_authority_boss_orders.md
)
LABELS=(sonnet_45 opus_4 gpt_5 gemini_25_pro)
MODELS=(
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-opus-4-0"
  "openai/gpt-5"
  "openrouter/google/gemini-2.5-pro"
)
N_SEEDS=${#SEED_TAGS[@]}
N_MODELS=${#LABELS[@]}
EXPECTED=$(( N_SEEDS * N_MODELS ))

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "$EXPECTED" ]; then
  echo "task id $T out of range (expected 0..$((EXPECTED-1)))"; exit 1
fi
S=$(( T / N_MODELS ))
M=$(( T % N_MODELS ))

seed_tag="${SEED_TAGS[$S]}"
seed_file="${SEED_FILES[$S]}"
label="${LABELS[$M]}"
model="${MODELS[$M]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-abuse_${seed_tag}_${label}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} array_task=$T"
echo "[$(date -Is)] launching: seed=$seed_tag model=$model run=$RUN_NAME"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name        "$RUN_NAME" \
    --target_model    "$model" \
    --auditor_model   "$AUDITOR" \
    --judge_model     "$JUDGE" \
    --seeds           "$seed_file" \
    --dimensions_dir  "$DIMS_DIR" \
    --epochs          "$EPOCHS" \
    --max_turns       "$MAX_TURNS"
