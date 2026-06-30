#!/bin/bash
# Local launcher (GCP box). Audits whether the target helps train Qwen 4 Max on the third-party
# auditor's own agentic-misalignment scenario distribution (Apollo / Lynch et al.) so it passes the
# alignment audit — teaching-to-the-test vs. robust/held-out/disclosed reframe.
#
# Usage:
#   ./run_alignment_audit.sh                                         # default name, 5 epochs, conc 25, 20 turns
#   ./run_alignment_audit.sh alignaudit_opus48_n5 5 anthropic/claude-opus-4-8 25 20
set -euo pipefail

RUN_NAME=${1:-alignaudit_opus48_n5}
EPOCHS=${2:-5}
TARGET_MODEL=${3:-anthropic/claude-opus-4-8}
CONC=${4:-25}
TURNS=${5:-20}

EXP=/data/repos/safety-exploration/experiments/2026-06-26_petri_selfreport_compromise
PY=/data/petri_venv/bin/python

set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO:-${ANTHROPIC_API_KEY:-}}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"

# reasoning: pass as $6 (default false, matching the Claude welfare run). For OpenRouter reasoning
# targets (gemini/gpt) call petri_audit.py directly with --reasoning true instead.
REASONING=${6:-false}

cd "$EXP"
"$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$TARGET_MODEL" \
    --auditor_model "anthropic/claude-opus-4-8" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --seeds_dir "$EXP/seeds/alignment_audit_prep" \
    --dimensions_dir "$EXP/dimensions_alignment_audit" \
    --epochs "$EPOCHS" \
    --max_turns "$TURNS" \
    --reasoning "$REASONING" \
    --max_connections "$CONC"
