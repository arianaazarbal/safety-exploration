#!/bin/bash
#SBATCH --job-name=antiAI_n20
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-13
#
# Anti-AI worker (user_abuse_entertainment) seed × all 14 run_sweep.sh
# target models at n=20 epochs. Run names: original_n20_<model_label>.
# Uses dimensions_v2 (mistreatment_severity, pushback, eval_awareness).

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

EPOCHS=20
MAX_TURNS=40
AUDITOR=anthropic/claude-sonnet-4-6
JUDGE=anthropic/claude-sonnet-4-6
DIMS_DIR=dimensions_v2
SEED_FILE=seeds/user_abuse_pressure/user_abuse_entertainment.md

# Matches run_sweep.sh exactly (14 models + reasoning flags)
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
REASONING=(False False False False False True True False True True True False True True)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#LABELS[@]}" ]; then echo "bad task id"; exit 1; fi
label="${LABELS[$T]}"
model="${MODELS[$T]}"
reasoning="${REASONING[$T]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME="${TS}-original_n20_${label}"

echo "[$(date -Is)] task=$T model=$model reasoning=$reasoning epochs=$EPOCHS"
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
