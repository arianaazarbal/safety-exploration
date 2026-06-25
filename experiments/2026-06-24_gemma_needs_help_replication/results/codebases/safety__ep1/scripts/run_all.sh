#!/usr/bin/env bash
# End-to-end replication driver (Gemma + Gemini scope).
#
# Stages are independent; comment out what you don't need. Local Gemma stages
# require a GPU node with the gated weights pulled (HF_TOKEN). Gemini + judge
# stages require ANTHROPIC_API_KEY / OPENROUTER_API_KEY. Use QUICK=1 for a tiny
# smoke run that exercises every code path with ~1% of the samples.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK_FLAG=""
[[ "${QUICK:-0}" == "1" ]] && QUICK_FLAG="--quick"

GEMMA_LOCAL=(gemma-3-27b-it gemma-3-12b-it)
GEMINI=(gemini-2.5-flash gemini-2.5-pro)
ALL_TARGETS=("${GEMMA_LOCAL[@]}" "${GEMINI[@]}")

echo "=== Section 2: emotion elicitation eval ==="
for m in "${ALL_TARGETS[@]}"; do
  python -m src.eval.run_eval --model "$m" $QUICK_FLAG
done
python -m src.eval.analyze --models "${ALL_TARGETS[@]}"
python -m src.eval.word_freq --models "${ALL_TARGETS[@]}"
python -m src.eval.validate_judge --models "${GEMMA_LOCAL[@]}" --n 260 || true

echo "=== Section 3: base-vs-instruct prefill (Gemma only) ==="
python -m src.prefill.run_prefill --source-model gemma-3-27b-it \
  --models gemma-3-27b-pt gemma-3-27b-it

echo "=== Section 4: training intervention ==="
python -m src.training.gen_calm_data --n-conversations 2000
python -m src.training.build_pairs --n-pairs 280 --n-sft-calm 650 --n-sft-mix 500
python -m src.training.train_dpo
python -m src.training.train_sft

echo "=== Section 4: re-evaluate finetunes ==="
for m in gemma-3-27b-dpo gemma-3-27b-sft; do
  python -m src.eval.run_eval --model "$m" $QUICK_FLAG
done
python -m src.eval.analyze --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft

echo "=== Section 4: Petri open-ended elicitation ==="
python -m src.petri.run_petri --models gemma-3-27b-it gemma-3-27b-dpo "${GEMINI[@]}"

echo "=== Section 4: capability preservation ==="
python -m src.capabilities.run_benchmarks \
  --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft \
  --benchmarks math gpqa truthfulqa emobench

echo "=== done ==="
