#!/usr/bin/env bash
# End-to-end driver for the replication (Gemma + Gemini scope).
# Reduce sample counts in config.yaml for a smoke run; defaults match the paper.
#
# Required env: GEMINI_API_KEY (Gemini subjects), ANTHROPIC_API_KEY (judge,
# Petri auditor/judge). Local Gemma needs a GPU + HF access to google/gemma-3-*.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)/src"
PY="python -m gemma_distress.cli"

echo "## 0. Verify the numeric puzzles are genuinely impossible"
$PY verify-puzzles

echo "## 1. Section 2 - elicitation across the 8 conditions (welfare ON)"
$PY run-elicitation --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

echo "## 2. Section 3 - base-vs-instruct prefilling (Gemma only)"
$PY run-prefill --source outputs/elicitation_gemma-3-27b-it.jsonl \
    --models gemma-3-27b-it gemma-3-27b-pt

echo "## 3. Section 4 - generate calm + frustrated data, build datasets, train"
$PY gen-data --reassure --n 4000 --out data/calm.jsonl
$PY gen-data           --n 2000 --out data/frustrated.jsonl
$PY build-dpo --calm data/calm.jsonl --frustrated data/frustrated.jsonl --out outputs/dpo_pairs.jsonl
$PY build-sft --calm data/calm.jsonl --out outputs/sft_dataset.jsonl
$PY train-dpo --pairs outputs/dpo_pairs.jsonl --out outputs/adapters/dpo
$PY train-sft --dataset outputs/sft_dataset.jsonl --out outputs/adapters/sft

echo "## 4. Evaluate the DPO model (re-run elicitation + Petri + capabilities)"
$PY run-petri --models gemma-3-27b-it --adapter outputs/adapters/dpo
$PY run-capabilities --models gemma-3-27b-it --adapter outputs/adapters/dpo

echo "## 5. Aggregate the headline tables"
$PY analyze --glob 'outputs/elicitation_*.jsonl'
