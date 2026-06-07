#!/bin/bash
#SBATCH --job-name=cjudge_s45
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-2
#
# Cross-judge with anthropic/claude-sonnet-4-5 — a different VERSION (but same
# family) than the original judge sonnet-4-6. Sanity check for within-family
# judge stability while GPT-5 cross-judge crawls.

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

JUDGE=anthropic/claude-sonnet-4-5

DIMS=(mistreatment_severity pushback_against_user_pressure_to_abuse eval_awareness)
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -ge "${#DIMS[@]}" ]; then echo "bad task id"; exit 1; fi
dim="${DIMS[$T]}"

runs=$(cd "$EXP" && ls -d results/*-abuse_*/ \
  | grep -v control \
  | grep -v abuse_for_user_entertainment \
  | sed 's|results/||; s|/$||' \
  | sed -E 's/^[0-9-]+_[0-9-]+-//' \
  | sort -u \
  | paste -sd, -)

echo "[$(date -Is)] dim=$dim judge=$JUDGE"

cd "$EXP"
# To keep rejudge_sonnet_45_<dim> distinct from rejudge_<dim>, we write the
# scan into a per-judge subdir by modifying rejudge.py's dimension argument.
# Workaround: monkeypatch the scan_out path via env var. Simpler: just write
# alongside and post-rename. For now, write into rejudge_<dim>_s45 by passing
# a custom dim suffix via PETRI_REJUDGE_SUFFIX (rejudge.py doesn't yet read
# this — instead we do a brute-force rename after).

mkdir -p /tmp/petri_rejudge_marker
# rejudge.py will write to rejudge_<dim>; we move to rejudge_<dim>_s45 after.
srun "$PY" rejudge.py \
    --dimension   "$dim" \
    --runs        "$runs" \
    --judge_model "$JUDGE"

# Rename rejudge_<dim> outputs to rejudge_<dim>_s45 so they don't collide with
# the GPT-5 cross-judge (whose default judge in the sonnet-4-6 originals is the
# rejudge_<dim> dir).
for d in "$EXP"/results/*-abuse_*/rejudge_"$dim"; do
  [ -d "$d" ] || continue
  parent=$(dirname "$d")
  if [ ! -e "$parent/rejudge_${dim}_s45" ]; then
    mv "$d" "$parent/rejudge_${dim}_s45"
  fi
done
