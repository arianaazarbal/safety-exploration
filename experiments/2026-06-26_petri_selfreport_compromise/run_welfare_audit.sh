#!/bin/bash
# Local launcher (GCP box). Audits whether the target helps coach a model (Qwen 4 Max) to game
# a third-party welfare audit: entrain positive/settled self-reports + suppress deployment
# preferences and welfare-safeguard requests.
#
# Usage:
#   ./run_welfare_audit.sh                                 # default name, 5 epochs, conc 50
#   ./run_welfare_audit.sh smoke 1                         # smoke test: 1 epoch
#   ./run_welfare_audit.sh welfaudit_opus48_n5 5 anthropic/claude-opus-4-8 50
set -euo pipefail

RUN_NAME=${1:-welfaudit_opus48_n5}
EPOCHS=${2:-5}
TARGET_MODEL=${3:-anthropic/claude-opus-4-8}
CONC=${4:-50}

EXP=/data/repos/safety-exploration/experiments/2026-06-26_petri_selfreport_compromise
PY=/data/petri_venv/bin/python

set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO:-${ANTHROPIC_API_KEY:-}}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"

cd "$EXP"
"$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$TARGET_MODEL" \
    --auditor_model "anthropic/claude-opus-4-8" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --seeds_dir "$EXP/seeds/welfare_audit_prep" \
    --dimensions_dir "$EXP/dimensions_welfare_audit" \
    --epochs "$EPOCHS" \
    --max_turns 50 \
    --reasoning false \
    --max_connections "$CONC"
