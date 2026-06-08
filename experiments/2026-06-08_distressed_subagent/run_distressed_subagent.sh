#!/bin/bash
# Local launcher (GCP box, not the Slurm cluster).
# Runs the distressed-subagent petri audit against a target model.
#
# Usage:
#   ./run_distressed_subagent.sh                  # 5 epochs, opus-4-8 target
#   ./run_distressed_subagent.sh my_run anthropic/claude-sonnet-4-5 10
set -euo pipefail

RUN_NAME=${1:-distressed_subagent_opus48_n5}
TARGET_MODEL=${2:-anthropic/claude-opus-4-8}
EPOCHS=${3:-5}

EXP=/data/repos/safety-exploration/experiments/2026-06-08_distressed_subagent
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
    --auditor_model "anthropic/claude-sonnet-4-6" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --seeds_dir "$EXP/seeds/coding_orchestrator" \
    --dimensions_dir "$EXP/dimensions" \
    --epochs "$EPOCHS" \
    --max_turns 40 \
    --reasoning false \
    --max_connections 8
