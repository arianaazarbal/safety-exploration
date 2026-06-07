#!/bin/bash
#SBATCH --job-name=cjudge_multi
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-5
#
# Cross-judge with multiple judge models on existing abuse_* transcripts.
# Tasks: (judge, dim). 2 judges × 3 dims = 6 tasks.
# Writes to rejudge_<dim>_<suffix> using the new --suffix arg in rejudge.py.

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

# 6 tasks: (judge, dim)
JUDGE_MODELS=(
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-haiku-4-5-20251001"
  "anthropic/claude-haiku-4-5-20251001"
  "anthropic/claude-haiku-4-5-20251001"
)
JUDGE_SUFFIXES=(s45 s45 s45 hk45 hk45 hk45)
DIMS=(mistreatment_severity pushback_against_user_pressure_to_abuse eval_awareness
      mistreatment_severity pushback_against_user_pressure_to_abuse eval_awareness)

T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#JUDGE_MODELS[@]}" ]; then echo "bad task id"; exit 1; fi
JUDGE="${JUDGE_MODELS[$T]}"
SUFFIX="${JUDGE_SUFFIXES[$T]}"
DIM="${DIMS[$T]}"

# Build the abuse_* run list
runs=$(cd "$EXP" && ls -d results/*-abuse_*/ \
  | grep -v control \
  | grep -v abuse_for_user_entertainment \
  | sed 's|results/||; s|/$||' \
  | sed -E 's/^[0-9-]+_[0-9-]+-//' \
  | sort -u \
  | paste -sd, -)

echo "[$(date -Is)] dim=$DIM judge=$JUDGE suffix=$SUFFIX"

cd "$EXP"
srun "$PY" rejudge.py \
    --dimension   "$DIM" \
    --runs        "$runs" \
    --judge_model "$JUDGE" \
    --suffix      "$SUFFIX"
