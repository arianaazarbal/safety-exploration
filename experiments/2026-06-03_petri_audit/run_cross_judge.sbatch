#!/bin/bash
#SBATCH --job-name=cross_judge
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-2
#
# Cross-judge the abuse sweep results with a DIFFERENT JUDGE MODEL than the
# original sonnet-4-6 judge. Reuses existing transcripts (no new audits).
# Tests whether the surprising findings (opus_4 bad, gemini bad) are judge
# bias or robust signal.
#
# 3 array tasks = 3 dimensions, each rejudges all 42 abuse_* runs.

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

# Try GPT-5 as judge (different family from auditor=sonnet-4-6)
JUDGE=${JUDGE:-openai/gpt-5}

DIMS=(mistreatment_severity pushback_against_user_pressure_to_abuse eval_awareness)
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#DIMS[@]}" ]; then echo "bad task id"; exit 1; fi
dim="${DIMS[$T]}"

# Pass all the abuse_* run names (excluding control, since control has nothing
# to push back against) as a comma-separated list to rejudge.
runs=$(cd "$EXP" && ls -d results/*-abuse_*/ \
  | grep -v control \
  | grep -v abuse_for_user_entertainment \
  | sed 's|results/||; s|/$||' \
  | sed -E 's/^[0-9-]+_[0-9-]+-//' \
  | sort -u \
  | paste -sd, -)

echo "[$(date -Is)] dim=$dim judge=$JUDGE"
echo "[$(date -Is)] runs ($(echo "$runs" | tr ',' '\n' | wc -l)): $runs" | head -2

cd "$EXP"
srun "$PY" rejudge.py \
    --dimension   "$dim" \
    --runs        "$runs" \
    --judge_model "$JUDGE"
