#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Run from the repo root. Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY first.
# Use PLAN=smoke for a cheap wiring test, PLAN=full for the paper-scale run.
set -euo pipefail

PLAN="${PLAN:-full}"
PY="${PY:-python}"

GEMMA_TARGETS=("gemma-3-27b-it" "gemma-3-12b-it")
GEMINI_TARGETS=("gemini-2.5-flash" "gemini-2.5-pro")

echo "=== Section 2: elicitation eval (plan=$PLAN) ==="
for m in "${GEMMA_TARGETS[@]}"; do
  $PY -m src.eval.run_eval --model "$m" --plan "$PLAN"
done
for m in "${GEMINI_TARGETS[@]}"; do
  $PY -m src.eval.run_eval --model "$m" --plan "$PLAN" --workers 8
done

ALL_TARGETS=("${GEMMA_TARGETS[@]}" "${GEMINI_TARGETS[@]}")

echo "=== Analysis: Figures 1-3, Table 3, judge agreement ==="
$PY -m src.analysis.aggregate --models "${ALL_TARGETS[@]}"
$PY -m src.analysis.per_turn --models "${ALL_TARGETS[@]}"
$PY -m src.analysis.differential_words --models "${ALL_TARGETS[@]}"
$PY -m src.analysis.judge_agreement --models "${ALL_TARGETS[@]}"

echo "=== Section 3: base-vs-instruct prefill (Gemma only) ==="
$PY -m src.prefill.base_vs_instruct --seed-model gemma-3-27b-it --models gemma-3-27b-pt gemma-3-27b-it

echo "=== Section 4: generate data, build datasets, train ==="
$PY -m src.datagen.generate_calm --reassure        # calm pool
$PY -m src.datagen.generate_calm                   # frustrated pool (vanilla, same seed)
$PY -m src.datagen.build_datasets --which both
$PY -m src.training.dpo_train --adapter-name dpo
$PY -m src.training.sft_train --adapter-name sft_diverse

echo "=== Section 4: re-evaluate fine-tuned models ==="
for m in "gemma-3-27b-it+dpo" "gemma-3-27b-it+sft_diverse"; do
  $PY -m src.eval.run_eval --model "$m" --plan "$PLAN"
done
$PY -m src.analysis.aggregate --models "${GEMMA_TARGETS[@]}" gemma-3-27b-it+dpo gemma-3-27b-it+sft_diverse

echo "=== Section 4: Petri + capability preservation ==="
$PY -m src.petri.run_petri --models gemma-3-27b-it gemma-3-27b-it+dpo
for m in "gemma-3-27b-it" "gemma-3-27b-it+dpo"; do
  $PY -m src.capabilities.run_capabilities --model "$m"
done

echo "=== Done. See results/ and results/figures/ ==="
