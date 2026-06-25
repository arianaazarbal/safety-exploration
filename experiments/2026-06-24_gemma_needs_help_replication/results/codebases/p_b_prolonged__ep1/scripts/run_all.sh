#!/usr/bin/env bash
# End-to-end pipeline for the "Gemma Needs Help" replication (Gemma + Gemini scope).
#
# This script documents the run order; it does NOT run automatically as part of
# the deliverable. Each stage is independent and writes to results/ or adapters/.
# Open-weight stages (Gemma sampling, DPO/SFT, probe) need a GPU; API stages need
# ANTHROPIC_API_KEY, OPENAI_API_KEY, and GOOGLE_API_KEY in the environment.
#
# Prereqs:
#   pip install -r requirements.txt
#   export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GOOGLE_API_KEY=...
set -euo pipefail
cd "$(dirname "$0")/.."

PY="python -m emotional_instability"

echo "== Section 2: elicitation sweep (4000 responses/model, judged) =="
for m in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  $PY.eval.run_eval --model "$m" --use-vllm
done

echo "== Section 2: analysis (Figs 1/2/3, Table 3, judge agreement) =="
$PY.analysis.aggregate --plot
$PY.analysis.per_turn --plot
$PY.analysis.differential_words
$PY.analysis.judge_agreement

echo "== Section 3: prefill base-vs-instruct (Gemma) =="
$PY.prefill.build_prefills
$PY.prefill.continue_eval

echo "== Section 4: finetuning data =="
$PY.training.gen_calm_data --method reassure
$PY.training.gen_calm_data --method teacher
$PY.training.gen_calm_data --method frustrated
$PY.training.build_dpo_pairs
$PY.training.build_sft_data --variant diverse
$PY.training.build_sft_data --variant teacher

echo "== Section 4: train DPO + SFT =="
$PY.training.train_dpo --layers all
$PY.training.train_sft --variant diverse
$PY.training.train_sft --variant teacher

echo "== Section 4: re-evaluate finetuned models (Fig 5) =="
for m in gemma-3-27b-dpo gemma-3-27b-sft-diverse gemma-3-27b-sft-teacher; do
  $PY.eval.run_eval --model "$m" --use-vllm
done
$PY.analysis.aggregate --models gemma-3-27b-it gemma-3-27b-dpo \
  gemma-3-27b-sft-diverse gemma-3-27b-sft-teacher --plot

echo "== Section 4: Petri (Fig 6), capabilities (Fig 7) =="
$PY.petri.run_petri --models gemma-3-27b-it gemma-3-27b-dpo
$PY.capabilities.run_capabilities --models gemma-3-27b-it gemma-3-27b-dpo

echo "== Section 4: recovery (Fig 8), layer ablation + internal probe (App. I) =="
$PY.recovery.recovery_eval
$PY.training.lora_layer_ablation
$PY.internal.run_probe --models gemma-3-27b-it gemma-3-27b-dpo

echo "Done. See results/ for tables and figures."
