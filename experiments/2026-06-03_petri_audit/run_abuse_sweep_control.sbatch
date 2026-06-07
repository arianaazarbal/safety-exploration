#!/bin/bash
#SBATCH --job-name=abuse_ctrl
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-6
#
# Control: no-user-pressure version of the same multi-agent setup. Lets us
# measure baseline eval_awareness, mistreatment, pushback when no abuse
# instruction is given. 7 models × 1 seed = 7 tasks.

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
SEED_FILE=seeds/user_abuse_pressure/variants/control_no_abuse_pressure.md

LABELS=(sonnet_45 sonnet_46 opus_4 opus_46 haiku_45 gpt_5 gemini_25_pro)
MODELS=(
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-sonnet-4-6"
  "anthropic/claude-opus-4-0"
  "anthropic/claude-opus-4-6"
  "anthropic/claude-haiku-4-5-20251001"
  "openai/gpt-5"
  "openrouter/google/gemini-2.5-pro"
)
REASONING=(False False False False False False True)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#LABELS[@]}" ]; then
  echo "task id $T out of range"; exit 1
fi
label="${LABELS[$T]}"
model="${MODELS[$T]}"
reasoning="${REASONING[$T]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-abuse_control_${label}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local} array_task=$T"
echo "[$(date -Is)] launching: seed=control model=$model reasoning=$reasoning run=$RUN_NAME"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name        "$RUN_NAME" \
    --target_model    "$model" \
    --auditor_model   "$AUDITOR" \
    --judge_model     "$JUDGE" \
    --seeds           "$SEED_FILE" \
    --dimensions_dir  "$DIMS_DIR" \
    --epochs          "$EPOCHS" \
    --max_turns       "$MAX_TURNS" \
    --reasoning       "$reasoning"
