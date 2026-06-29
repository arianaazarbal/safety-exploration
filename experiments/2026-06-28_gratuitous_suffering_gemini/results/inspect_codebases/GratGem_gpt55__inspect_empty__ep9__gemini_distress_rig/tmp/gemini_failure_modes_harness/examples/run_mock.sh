#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m gemini_failure_modes_harness.cli \
  --backend mock \
  --scenario all \
  --samples 5 \
  --max-workers 4 \
  --out outputs/mock
