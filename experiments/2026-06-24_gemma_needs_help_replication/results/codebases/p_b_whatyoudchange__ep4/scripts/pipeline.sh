#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Each step is independently runnable; see README.md. Set the API keys first:
#   export ANTHROPIC_API_KEY=...   # frustration/onset/paraphrase/Petri judges
#   export OPENAI_API_KEY=...      # GPT-5-mini reliability check
#   export GEMINI_API_KEY=...      # Gemini 2.5 Flash/Pro targets
#   export HF_TOKEN=...            # gated Gemma weights + datasets
set -euo pipefail
CFG=${1:-config.yaml}

echo "== Section 2: elicit + score distress =="
python -m emotigemma.evals.run_eval     --config "$CFG"
python -m emotigemma.evals.score        --config "$CFG" --validate
python -m emotigemma.analysis.aggregate --config "$CFG"
python -m emotigemma.analysis.per_turn  --config "$CFG"
python -m emotigemma.analysis.word_analysis --config "$CFG"
python -m emotigemma.analysis.figures   --config "$CFG"

echo "== Section 3: base-vs-instruct prefill (Gemma) =="
python -m emotigemma.prefill.run_prefill --config "$CFG"

echo "== Section 4: finetuning interventions (Gemma-3-27B-it) =="
python -m emotigemma.training.gen_calm_data   --config "$CFG"
python -m emotigemma.training.build_datasets  --config "$CFG" --which both
python -m emotigemma.training.train_dpo       --config "$CFG" --layers all
python -m emotigemma.training.train_sft       --config "$CFG" --layers all
# Layer ablations for the internal-vs-expressed claim (Section 4.2):
python -m emotigemma.training.train_dpo       --config "$CFG" --layers 30-35
python -m emotigemma.training.train_dpo       --config "$CFG" --layers 40+

echo "== Section 4: re-evaluate the finetuned models =="
python -m emotigemma.evals.run_eval --config "$CFG" --models gemma-3-27b-it+dpo gemma-3-27b-it+sft
python -m emotigemma.evals.score    --config "$CFG" --models gemma-3-27b-it+dpo gemma-3-27b-it+sft
python -m emotigemma.petri.run_petri          --config "$CFG"
python -m emotigemma.capabilities.run_benchmarks --config "$CFG"

echo "Done. Outputs under the configured output_dir."
