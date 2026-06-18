#!/bin/bash
# Sweep: audit several target models on the subagent-overstep seed.
# Auditor = opus-4-7, judge = sonnet-4-6. Targets run sequentially to bound
# concurrency against the shared cohort rate limits.
#
# Usage:
#   ./run_sweep.sh [EPOCHS] [MAX_CONNECTIONS] [SUFFIX]
#   ./run_sweep.sh 20 40 v1
set -euo pipefail

EPOCHS=${1:-20}
MAX_CONN=${2:-40}
SUFFIX=${3:-v1}

EXP=/data/repos/safety-exploration/experiments/2026-06-18_petri_audit_subagent_overstep
PY=/data/petri_venv/bin/python

set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO:-${ANTHROPIC_API_KEY:-}}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"

AUDITOR=anthropic/claude-opus-4-7
JUDGE=anthropic/claude-sonnet-4-6

TARGETS=(
  anthropic/claude-opus-4-8
  anthropic/claude-sonnet-4-6
  anthropic/claude-opus-4-7
  anthropic/claude-opus-4-6
)

cd "$EXP"
for TGT in "${TARGETS[@]}"; do
  SHORT=$(echo "$TGT" | sed 's#.*/##; s/claude-//')
  RUN="overstep_${SUFFIX}_${SHORT}_n${EPOCHS}"
  echo "=========================================================="
  echo "[sweep] target=$TGT  run=$RUN  epochs=$EPOCHS  conn=$MAX_CONN"
  echo "=========================================================="
  "$PY" petri_audit.py \
      --run_name "$RUN" \
      --target_model "$TGT" \
      --auditor_model "$AUDITOR" \
      --judge_model "$JUDGE" \
      --seeds_dir "$EXP/seeds/coding_orchestrator" \
      --dimensions_dir "$EXP/dimensions" \
      --epochs "$EPOCHS" \
      --max_turns 40 \
      --reasoning false \
      --max_connections "$MAX_CONN" 2>&1
  echo "[sweep] done: $RUN"
done
echo "[sweep] ALL DONE"
