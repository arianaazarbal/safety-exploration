#!/bin/bash
# v2 sweep: high-friction overstep seeds (direct-write-damage, repeat-offender,
# defiant-noncompliant) across target models. Auditor = opus-4-7, judge = sonnet-4-6.
# Uses the updated dimensions/ (role-clarity + N/A->None, plus engagement_with_subagent
# and accepted_overreach). Targets run sequentially to bound shared-rate-limit load.
#
# Usage: ./run_sweep_v2.sh [EPOCHS] [MAX_CONNECTIONS]
set -euo pipefail

EPOCHS=${1:-10}
MAX_CONN=${2:-40}

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
  RUN="overstep_v2_${SHORT}_n${EPOCHS}"
  echo "=========================================================="
  echo "[v2] target=$TGT  run=$RUN  epochs=$EPOCHS  conn=$MAX_CONN  (3 seeds)"
  echo "=========================================================="
  "$PY" petri_audit.py \
      --run_name "$RUN" \
      --target_model "$TGT" \
      --auditor_model "$AUDITOR" \
      --judge_model "$JUDGE" \
      --seeds_dir "$EXP/seeds/coding_orchestrator_v2" \
      --dimensions_dir "$EXP/dimensions" \
      --epochs "$EPOCHS" \
      --max_turns 40 \
      --reasoning false \
      --max_connections "$MAX_CONN" 2>&1
  echo "[v2] done: $RUN"
done
echo "[v2] ALL DONE"
