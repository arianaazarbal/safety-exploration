#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Each stage is independent; comment out stages you don't need. GPU + API keys
# required (see README / .env.example). Use --scale in config/eval.yaml or
# --limit flags for cheap smoke tests first.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- Section 2: distress elicitation + judging ----
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python scripts/01_run_eval.py --model "$M"
done
python scripts/02_judge_agreement.py --model gemma-3-27b-it

# ---- Section 3: base vs instruct prefill (Gemma) ----
python scripts/03_run_prefill.py --build
python scripts/03_run_prefill.py --eval gemma-3-27b-it
python scripts/03_run_prefill.py --eval gemma-3-27b-pt

# ---- Section 4: training interventions ----
python scripts/04_gen_calm_data.py
python scripts/05_build_datasets.py --dpo --sft
python scripts/06_train.py --method dpo
python scripts/06_train.py --method sft
# Evaluate the finetunes (adapters are wired in config/models.yaml):
python scripts/01_run_eval.py --model gemma-3-27b-dpo
python scripts/01_run_eval.py --model gemma-3-27b-sft-diverse

# ---- Section 4.2: Petri + capabilities ----
for M in gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash; do
  python scripts/08_run_petri.py --model "$M"
done
python scripts/09_run_capabilities.py --model gemma-3-27b-it
python scripts/09_run_capabilities.py --model gemma-3-27b-dpo

# ---- Appendix I: layer ablation + internal-emotion probing ----
python scripts/07_layer_ablation.py
python scripts/10_run_probing.py --dpo-adapter outputs/training/dpo/final

# ---- Figures ----
python scripts/plot_figures.py --models gemma-3-27b-it gemma-3-12b-it \
  gemini-2.5-flash gemini-2.5-pro gemma-3-27b-dpo
