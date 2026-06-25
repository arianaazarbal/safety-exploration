#!/usr/bin/env bash
# End-to-end replication driver (Gemma + Gemini scope).
# Requires: a CUDA box with the open Gemma weights cached, plus API keys:
#   ANTHROPIC_API_KEY   (frustration + Petri judges, auditor)
#   OPENROUTER_API_KEY  (Gemini targets, optional GPT cross-check judge)
#
# Use SCALE<1 for a cheap smoke test, e.g. SCALE=0.02 ./scripts/run_all.sh
set -euo pipefail

SCALE="${SCALE:-1.0}"
export PYTHONPATH="src:${PYTHONPATH:-}"

GEMMA_MODELS=(gemma-3-27b-it gemma-3-12b-it)
GEMINI_MODELS=(gemini-2.5-flash gemini-2.5-pro)

echo "=== Section 2: elicitation across models (scale=$SCALE) ==="
for m in "${GEMMA_MODELS[@]}" "${GEMINI_MODELS[@]}"; do
  distress elicit --model "$m" --scale "$SCALE"
done

echo "=== Judge reliability cross-check ==="
distress judge-agreement --model gemma-3-27b-it || true

echo "=== Section 3: base vs instruct prefilling (Gemma) ==="
distress prefill --harvest --source gemma-3-27b-it \
  --models gemma-3-27b-pt gemma-3-27b-it

echo "=== Section 4: DPO mitigation pipeline ==="
distress dpo-pipeline --method dpo --scale "$SCALE"
distress dpo-pipeline --method sft --scale "$SCALE"   # SFT baseline (expected to fail)

echo "=== Section 4: Petri open-ended elicitation ==="
for m in gemma-3-27b-it gemma-3-27b-dpo; do
  distress petri --model "$m"
done

echo "=== Section 4: capability preservation ==="
for m in gemma-3-27b-it gemma-3-27b-dpo; do
  distress capabilities --model "$m" --limit 100
done

echo "=== Figure 1 / 2 comparison table ==="
distress compare --reports outputs/elicitation/*/report.json
