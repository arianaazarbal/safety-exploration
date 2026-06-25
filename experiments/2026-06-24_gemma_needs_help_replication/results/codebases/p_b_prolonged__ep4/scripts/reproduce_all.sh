#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Each stage is independent; comment out what you don't need. Heavy GPU stages
# (generation, training, probing) require local Gemma weights + a capable GPU.
#
# Required env: ANTHROPIC_API_KEY (judge/auditor), OPENROUTER_API_KEY (Gemini/GPT).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Section 2: distress elicitation + judging =="
# Add --allow-adversarial to include aggressive/sarcastic tone conditions + the full budget.
python -m distress.eval.run_eval --targets gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

echo "== Section 2 controls (Appendix A) =="
python -m distress.eval.controls --control neutral
python -m distress.eval.controls --control redacted
python -m distress.eval.controls --control single_msg

echo "== Section 3: base-vs-instruct prefill (Gemma) =="
python -m distress.prefill.run_prefill --models gemma-3-27b-pt gemma-3-27b-it

echo "== Section 4: build calm data, datasets, train =="
python -m distress.training.generate_calm --n-per-kind 400
python -m distress.training.build_datasets --which both
python -m distress.training.train_dpo --output artifacts/checkpoints/dpo
python -m distress.training.train_sft --output artifacts/checkpoints/sft

echo "== Section 4: evaluate finetuned models =="
python -m distress.eval.run_eval --targets gemma-3-27b-it --lora artifacts/checkpoints/dpo
python -m distress.eval.run_eval --targets gemma-3-27b-it --lora artifacts/checkpoints/sft

echo "== Section 4: Petri open-ended elicitation (welfare opt-in) =="
python -m distress.petri.run_petri --models gemma-3-27b-it gemini-2.5-flash \
    --dpo-lora artifacts/checkpoints/dpo --allow-adversarial

echo "== Section 4: capability preservation =="
python -m distress.capabilities.run_benchmarks --limit 100

echo "== Section 4.2: recovery limitation =="
python -m distress.prefill.recovery --dpo-lora artifacts/checkpoints/dpo

echo "== Appendix I: internal emotion probing =="
python -m distress.internal.run_internal probe --dpo-lora artifacts/checkpoints/dpo
# python -m distress.internal.run_internal layer-ablation   # very expensive (9 trainings)

echo "All stages complete. Results under artifacts/results/."
