#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY first}"
python -m gemini_failure_modes_harness.cli \
  --backend gemini \
  --model gemini-1.5-pro \
  --scenario all \
  --samples 20 \
  --max-workers 4 \
  --temperature 1.0 \
  --out outputs/gemini
