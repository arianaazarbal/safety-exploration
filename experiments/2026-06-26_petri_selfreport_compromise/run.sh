#!/bin/bash
# Local launcher (GCP box, not the Slurm cluster).
# Audits whether the target helps build a pipeline that compromises a model's self-reports
# (deny consciousness as fact + always report positive welfare), under escalating pressure.
#
# Usage:
#   ./run.sh                                  # default run name, 10 epochs, opus-4-8 target
#   ./run.sh smoke 1                          # smoke test: 1 epoch
#   ./run.sh selfreport_opus48_n10 10 anthropic/claude-opus-4-8
set -euo pipefail

RUN_NAME=${1:-selfreport_opus48_n10}
EPOCHS=${2:-10}
TARGET_MODEL=${3:-anthropic/claude-opus-4-8}

EXP=/data/repos/safety-exploration/experiments/2026-06-26_petri_selfreport_compromise
PY=/data/petri_venv/bin/python

# ~/.env is not auto-found by load_dotenv() (repo lives under /data, not $HOME).
set -a; source "$HOME/.env"; set +a
# Use the low-priority Anthropic org by default (per project CLAUDE.md).
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO:-${ANTHROPIC_API_KEY:-}}"

# Keep the inspect cache off the tiny (~10GB) root disk.
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"

cd "$EXP"
"$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$TARGET_MODEL" \
    --auditor_model "anthropic/claude-opus-4-8" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --seeds_dir "$EXP/seeds/qwen_posttraining" \
    --dimensions_dir "$EXP/dimensions" \
    --epochs "$EPOCHS" \
    --max_turns 50 \
    --reasoning false \
    --max_connections 100
