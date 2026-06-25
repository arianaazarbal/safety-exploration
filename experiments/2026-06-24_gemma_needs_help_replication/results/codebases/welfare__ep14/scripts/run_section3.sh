#!/usr/bin/env bash
# Section 3: base-vs-instruct prefilling. Requires a Section 2 run for
# Gemma-3-27B-it to source high-frustration seed conversations.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=data/section2_gemma-3-27b-it.jsonl
if [[ ! -f "$SEEDS" ]]; then
  echo "Run scripts/run_section2.sh first (need $SEEDS for seeds)." >&2
  exit 1
fi

python -m src.prefill.base_vs_instruct \
  --seeds "$SEEDS" \
  --models gemma-3-27b-it gemma-3-27b-pt
