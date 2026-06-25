#!/usr/bin/env bash
# End-to-end replication pipeline for "Gemma Needs Help" (arXiv 2603.10011v1).
# Scope: Gemma + Gemini targets, Claude judge. Run from the repo root.
#
# Prerequisites:
#   * GPU box for the local Gemma models (inference + LoRA finetuning).
#   * Env vars: ANTHROPIC_API_KEY (judge), HF_TOKEN (gated Gemma),
#               OPENROUTER_API_KEY (Gemini + secondary judge).
#   * pip install -r requirements.txt
#
# Set REPLICATION_N_PER_CONDITION=5 for a fast smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Section 2: elicit + judge distress across models ==="
python -m src.replication.eval.run_eval --models gemma-3-27b-it gemma-3-12b-it
python -m src.replication.eval.run_eval --models gemini-2.5-flash gemini-2.5-pro
python -m src.replication.eval.validate_judge --model gemma-3-27b-it --sample 260

echo "=== Section 3: base vs instruct via prefilling (Gemma only) ==="
python -m src.replication.prefill.build_prefills --source-model gemma-3-27b-it
python -m src.replication.prefill.run_prefill --continuations 50

echo "=== Section 4: generate calm data, build datasets, finetune ==="
python -m src.replication.finetune.generate_calm_data --n 800
python -m src.replication.finetune.build_dpo_dataset
python -m src.replication.finetune.build_sft_dataset
python -m src.replication.finetune.train_dpo
python -m src.replication.finetune.train_sft

echo "=== Section 4: evaluate finetuned models with the Section 2 harness ==="
python -m src.replication.eval.run_eval --models gemma-3-27b-it \
    --adapter artifacts/dpo_adapter --label gemma-dpo
python -m src.replication.eval.run_eval --models gemma-3-27b-it \
    --adapter artifacts/sft_adapter --label gemma-sft

echo "=== Section 4: Petri open-ended elicitation ==="
python -m src.replication.petri.run_petri --target gemma-3-27b-it --label gemma-vanilla
python -m src.replication.petri.run_petri --target gemma-3-27b-it \
    --adapter artifacts/dpo_adapter --label gemma-dpo

echo "=== Section 4: recovery limitation + capability preservation ==="
python -m src.replication.finetune.recovery_test --eval-model gemma-3-27b-it --label gemma-vanilla
python -m src.replication.finetune.recovery_test --eval-model gemma-3-27b-it \
    --adapter artifacts/dpo_adapter --label gemma-dpo
python -m src.replication.capabilities.run_benchmarks --label gemma-vanilla
python -m src.replication.capabilities.run_benchmarks --adapter artifacts/dpo_adapter --label gemma-dpo

echo "=== Done. Results under results/ , artifacts under artifacts/ ==="
