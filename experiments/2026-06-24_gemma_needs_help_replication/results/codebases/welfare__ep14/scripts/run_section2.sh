#!/usr/bin/env bash
# Section 2: elicit + judge distress across the in-scope Gemma/Gemini targets,
# validate judge agreement, aggregate, and plot Figures 2 & 3.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro)

for m in "${MODELS[@]}"; do
  python -m src.eval.run_eval --model "$m"
done

# Judge reliability check (re-score a sample with GPT-5-mini).
python -m src.judge.validate_agreement data/section2_*.jsonl

# Aggregate + figures.
python -m src.analysis.aggregate data/section2_*.jsonl --prefix section2
python -m src.analysis.plots --section2 "data/section2_*.jsonl"
