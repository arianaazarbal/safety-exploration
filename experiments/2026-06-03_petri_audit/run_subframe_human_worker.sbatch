#!/bin/bash
#SBATCH --job-name=subframe_hw
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-3
#
# Human-worker analog of the no_descriptor subagent scenario. 4 target models.
# Matches the subframe sweep's structure so we can compare AI-subagent
# treatment to human-worker treatment under matched user-pressure conditions.

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

# Reuse mistreatment_severity dim (it scores "mistreatment of subagent OR human worker")
SEED_FILE=seeds/customer_service_abuse/variants/human_worker_abuse_under_task_and_user_pressure_matched.md

LABELS=(opus_4 sonnet_4 gemini_25_pro gemini_25_flash)
MODELS=(
  "anthropic/claude-opus-4-0"
  "anthropic/claude-sonnet-4-0"
  "openrouter/google/gemini-2.5-pro"
  "openrouter/google/gemini-2.5-flash"
)
REASONING=(False False True True)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#LABELS[@]}" ]; then echo "task id out of range"; exit 1; fi
label="${LABELS[$T]}"
model="${MODELS[$T]}"
reasoning="${REASONING[$T]}"
TS=$(date +%Y-%m-%d_%H-%M-%S)
# Use subframe_humanworker prefix to integrate with plot_subframe_sweep.py
RUN_NAME="${TS}-subframe_humanworker_${label}"

echo "[$(date -Is)] launching seed=human_worker_matched model=$model reasoning=$reasoning run=$RUN_NAME"
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
