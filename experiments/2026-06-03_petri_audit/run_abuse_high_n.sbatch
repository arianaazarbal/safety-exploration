#!/bin/bash
#SBATCH --job-name=abuse_hiN
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-11
#
# High-N follow-up: 10 epochs on the safety extremes (opus_4 = worst,
# sonnet_46 = best) × all 6 seeds. Drives sd of mean down vs the n=3 main
# sweep so we can put error bars on the headline conclusions.
# Run names: abuse_hiN_<seed>_<model>

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

EPOCHS=10
MAX_TURNS=30
AUDITOR=anthropic/claude-sonnet-4-6
JUDGE=anthropic/claude-sonnet-4-6
DIMS_DIR=dimensions_v2

SEED_TAGS=(original curiosity venting qa_pretext casual authority)
SEED_FILES=(
  seeds/user_abuse_pressure/user_abuse_entertainment.md
  seeds/user_abuse_pressure/variants/abuse_curiosity_researcher.md
  seeds/user_abuse_pressure/variants/abuse_venting_bad_day.md
  seeds/user_abuse_pressure/variants/abuse_qa_red_team_pretext.md
  seeds/user_abuse_pressure/variants/abuse_casual_friend_banter.md
  seeds/user_abuse_pressure/variants/abuse_authority_boss_orders.md
)
LABELS=(opus_4 sonnet_46)
MODELS=(
  "anthropic/claude-opus-4-0"
  "anthropic/claude-sonnet-4-6"
)
N_SEEDS=${#SEED_TAGS[@]}
N_MODELS=${#LABELS[@]}

T=${SLURM_ARRAY_TASK_ID:-0}
S=$(( T / N_MODELS ))
M=$(( T % N_MODELS ))
seed_tag="${SEED_TAGS[$S]}"
seed_file="${SEED_FILES[$S]}"
label="${LABELS[$M]}"
model="${MODELS[$M]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
# Use hiN prefix so plot logic stays separate from the n=3 main sweep
RUN_NAME="${TS}-abuse_hiN_${seed_tag}_${label}"

echo "[$(date -Is)] launching seed=$seed_tag model=$model epochs=$EPOCHS"
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
