#!/usr/bin/env bash
# Section 2: elicit & quantify distress across Gemma + Gemini models, then build
# Figures 1-3. Requires ANTHROPIC_API_KEY (judge) and OPENROUTER_API_KEY (Gemini).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("gemma-3-27b-it" "gemma-3-12b-it" "gemini-2.5-flash" "gemini-2.5-pro")

for m in "${MODELS[@]}"; do
  echo "=== Section 2: $m ==="
  python -m emotional_instability.run_eval --model "$m"
done

python -m emotional_instability.analyze
