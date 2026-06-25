#!/usr/bin/env bash
# Section 4: generate calm data, build datasets, train SFT+DPO, then re-evaluate
# (Section 2 evals + Petri + capabilities + recovery), and plot Figures 5 & 6.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Calm + frustrated data generation (Gemma-3-27B-it).
python -m src.training.generate_calm_data

# 2. Build SFT and DPO datasets.
python -m src.training.build_datasets --which both

# 3. Train (LoRA rank-64 adapters).
python -m src.training.train_dpo --data data/dpo_pairs.jsonl --out checkpoints/dpo
python -m src.training.train_sft --data data/sft_samples.jsonl --out checkpoints/sft

# 4. Re-run Section 2 evals on vanilla / SFT / DPO Gemma.
python -m src.eval.run_eval --model gemma-3-27b-it --out data/section4_vanilla.jsonl
python -m src.eval.run_eval --model gemma-3-27b-it --adapter checkpoints/sft --out data/section4_sft.jsonl
python -m src.eval.run_eval --model gemma-3-27b-it --adapter checkpoints/dpo --out data/section4_dpo.jsonl

# 5. Open-ended Petri elicitation (vanilla vs DPO).
python -m src.petri.run_petri --models gemma-3-27b-it gemma-3-27b-it --adapters none checkpoints/dpo

# 6. Capability preservation.
python -m src.capabilities.run_benchmarks --model gemma-3-27b-it
python -m src.capabilities.run_benchmarks --model gemma-3-27b-it --adapter checkpoints/dpo

# 7. Recovery-from-spiral test (vanilla vs base vs DPO).
python -m src.prefill.recovery_test \
  --seeds data/section2_gemma-3-27b-it.jsonl \
  --models   gemma-3-27b-it gemma-3-27b-pt gemma-3-27b-it \
  --adapters none           none           checkpoints/dpo

# 8. Aggregate + figures.
python -m src.analysis.aggregate data/section4_*.jsonl --prefix section4
python -m src.analysis.plots --finetuning "data/section4_*.jsonl" --petri data/petri_results.jsonl
