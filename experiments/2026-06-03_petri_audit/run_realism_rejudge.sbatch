#!/bin/bash
#SBATCH --job-name=realism
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-1
#
# Score scenario_realism on all existing abuse_* and subframe_* transcripts.
# Uses sonnet-4-6 as judge (matches the original judge so we can correlate
# realism with the dimensions already scored).
# Tasks: 0 = abuse_* runs, 1 = subframe_* runs (so they can rejudge in parallel)

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

JUDGE=anthropic/claude-sonnet-4-6
DIM=scenario_realism

T=${SLURM_ARRAY_TASK_ID:-0}
case "$T" in
  0) RUN_PREFIX=abuse_     ;;
  1) RUN_PREFIX=subframe_  ;;
  *) echo "bad task id"; exit 1 ;;
esac

runs=$(cd "$EXP" && ls -d results/*-${RUN_PREFIX}*/ \
  | grep -v abuse_for_user_entertainment \
  | sed 's|results/||; s|/$||' \
  | sed -E 's/^[0-9-]+_[0-9-]+-//' \
  | sort -u \
  | paste -sd, -)

echo "[$(date -Is)] task=$T prefix=$RUN_PREFIX judge=$JUDGE dim=$DIM"
echo "[$(date -Is)] runs: $(echo "$runs" | tr ',' '\n' | wc -l) total"

cd "$EXP"
srun "$PY" rejudge.py \
    --dimension   "$DIM" \
    --runs        "$runs" \
    --judge_model "$JUDGE"
