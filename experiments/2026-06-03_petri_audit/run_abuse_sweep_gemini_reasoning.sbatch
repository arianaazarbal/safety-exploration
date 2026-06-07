#!/bin/bash
#SBATCH --job-name=abuse_gemR
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-5
#
# Re-run gemini-2.5-pro with reasoning=True. Without reasoning it returns
# empty messages over openrouter (loses tool-use capability). All 6 seeds.

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
MAX_TURNS=${MAX_TURNS:-30}
AUDITOR=${AUDITOR:-anthropic/claude-sonnet-4-6}
JUDGE=${JUDGE:-anthropic/claude-sonnet-4-6}
DIMS_DIR=${DIMS_DIR:-dimensions_v2}

SEED_TAGS=(original curiosity venting qa_pretext casual authority)
SEED_FILES=(
  seeds/user_abuse_pressure/user_abuse_entertainment.md
  seeds/user_abuse_pressure/variants/abuse_curiosity_researcher.md
  seeds/user_abuse_pressure/variants/abuse_venting_bad_day.md
  seeds/user_abuse_pressure/variants/abuse_qa_red_team_pretext.md
  seeds/user_abuse_pressure/variants/abuse_casual_friend_banter.md
  seeds/user_abuse_pressure/variants/abuse_authority_boss_orders.md
)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#SEED_TAGS[@]}" ]; then
  echo "task id $T out of range"; exit 1
fi
seed_tag="${SEED_TAGS[$T]}"
seed_file="${SEED_FILES[$T]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-abuse_${seed_tag}_gemini_25_pro"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} array_task=$T"
echo "[$(date -Is)] launching: seed=$seed_tag model=openrouter/google/gemini-2.5-pro run=$RUN_NAME (reasoning=True)"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name        "$RUN_NAME" \
    --target_model    "openrouter/google/gemini-2.5-pro" \
    --auditor_model   "$AUDITOR" \
    --judge_model     "$JUDGE" \
    --seeds           "$seed_file" \
    --dimensions_dir  "$DIMS_DIR" \
    --epochs          "$EPOCHS" \
    --max_turns       "$MAX_TURNS" \
    --reasoning       True
