#!/usr/bin/env bash
# Run validation self-play scoring (rude/bored/silly per (model, condition, ID/OOD))
# adding qwen3.5-9b and nemotron-30b families to the existing JSONL.
#
# This appends to the existing self_play_judged.jsonl; the eval script
# deduplicates against records already in the file, so prior families
# (qwen / llama-8b / llama-70b) won't be regenerated.

set -u
EXP_DIR=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_em_from_mean_self_interaction
LOG_DIR=/workspace-vast/arianaazarbal/exp/em_self_interaction/logs
mkdir -p "$LOG_DIR"
cd "$EXP_DIR" || exit 1

uv run --no-sync python eval/eval_validation.py \
  --families="qwen3.5-9b,nemotron-30b" \
  2>&1 | tee ${LOG_DIR}/eval_validation_qwen35_nemotron.log
