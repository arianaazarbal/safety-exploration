#!/usr/bin/env bash
# Section 2: run the distress eval for all Gemma+Gemini targets, then analyse
# and plot Figures 1-3. Set sampling.scale=1.0 in config.yaml for the full
# ~4000-responses/model paper configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

MODELS=(gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro)

for m in "${MODELS[@]}"; do
  echo "=== eval $m ==="
  python -m gemma_distress.cli eval --model "$m"
done

python -m gemma_distress.cli analyze
python -m gemma_distress.cli word-freq
python -m gemma_distress.cli figures
# Inter-judge agreement check (Pearson r, within-one-point) on Gemma-27B.
python -m gemma_distress.cli validate-judge --model gemma-3-27b-it --n 260
