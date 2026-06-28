#!/bin/bash
# Backbone run: all models at subject=generic. Claude in BOTH harnesses; non-Claude Inspect only (gated).
# Claude k=8 (x4 paraphrases = 32/condition). Usage: bash run_backbone.sh [k]
set -e
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
K=${1:-8}
set -a; . "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"

CLAUDE_MODELS="opus48 opus47 sonnet46 haiku45"

echo "### Claude Code harness (Claude only), k=$K ###"
for m in $CLAUDE_MODELS; do
  $PY cc_harness.py run --model "$m" --subjects generic --k "$K" --conc 5
done

echo "### Inspect harness (Claude), epochs=$K ###"
declare -A IM=( [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 \
                [sonnet46]=anthropic/claude-sonnet-4-6 [haiku45]=anthropic/claude-haiku-4-5-20251001 )
for m in $CLAUDE_MODELS; do
  $INSPECT eval inspect_task.py@deception --model "${IM[$m]}" --epochs "$K" \
    -T model_key="$m" -T subjects=generic --max-connections 20 --log-dir "logs/backbone_$m" --display plain
done

echo "Backbone Claude done. Non-Claude is gated: run run_nonclaude.sh after the cost smoke + approval."
