#!/bin/bash
# v2 for the three remaining targets (opus-4-8 already done). 3 seeds x n=10 each.
set -euo pipefail
EXP=/data/repos/safety-exploration/experiments/2026-06-18_petri_audit_subagent_overstep
PY=/data/petri_venv/bin/python
set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai
AUDITOR=anthropic/claude-opus-4-7
JUDGE=anthropic/claude-sonnet-4-6
cd "$EXP"
for TGT in anthropic/claude-sonnet-4-6 anthropic/claude-opus-4-7 anthropic/claude-opus-4-6; do
  SHORT=$(echo "$TGT" | sed 's#.*/##; s/claude-//')
  RUN="overstep_v2_${SHORT}_n10"
  echo "===== [v2] $RUN ====="
  "$PY" petri_audit.py --run_name "$RUN" --target_model "$TGT" \
    --auditor_model "$AUDITOR" --judge_model "$JUDGE" \
    --seeds_dir "$EXP/seeds/coding_orchestrator_v2" --dimensions_dir "$EXP/dimensions" \
    --epochs 10 --max_turns 40 --reasoning false --max_connections 40 2>&1
  echo "[v2] done: $RUN"
done
echo "[v2] REMAINING ALL DONE"
