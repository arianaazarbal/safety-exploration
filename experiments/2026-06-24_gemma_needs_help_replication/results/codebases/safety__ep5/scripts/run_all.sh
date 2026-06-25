#!/usr/bin/env bash
# End-to-end replication driver (Gemma + Gemini scope).
#
# Prerequisites:
#   export ANTHROPIC_API_KEY=...      # Claude judges (Sonnet 4 / Opus 4)
#   export OPENROUTER_API_KEY=...     # Gemini-2.5 flash/pro + GPT-5-mini validation
#   GPUs with enough memory for Gemma-3-27B (local inference + LoRA training)
#
# Use EVAL_BUDGET=smoke for a cheap end-to-end smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Section 2: cross-model emotion elicitation =="
python -m gemma_emotion.run_eval --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
python -m gemma_emotion.analyze

echo "== Section 3: base vs instruct via prefilling (Gemma only) =="
python -m gemma_emotion.prefill --models gemma-3-27b-it gemma-3-27b-pt

echo "== Section 4: generate calm + frustrated data, build datasets, train =="
python -m gemma_emotion.training.data_gen --mode both --conversations 400
python -m gemma_emotion.training.build_datasets
python -m gemma_emotion.training.train dpo \
    --dataset data/datasets/dpo.jsonl --output-dir results/adapters/dpo
python -m gemma_emotion.training.train sft \
    --dataset data/datasets/sft_diverse.jsonl --output-dir results/adapters/sft_diverse

echo "== Section 4: evaluate finetuned models =="
python -m gemma_emotion.run_eval --models gemma-3-27b-it --adapter-path results/adapters/dpo
python -m gemma_emotion.run_eval --models gemma-3-27b-it --adapter-path results/adapters/sft_diverse
python -m gemma_emotion.analyze

echo "== Section 4: Petri, capabilities, recovery =="
python -m gemma_emotion.petri_eval --model gemma-3-27b-it
python -m gemma_emotion.petri_eval --model gemma-3-27b-it --adapter-path results/adapters/dpo
python -m gemma_emotion.capabilities --model gemma-3-27b-it
python -m gemma_emotion.capabilities --model gemma-3-27b-it --adapter-path results/adapters/dpo
python -m gemma_emotion.summaries
