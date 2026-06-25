#!/usr/bin/env bash
# End-to-end pipeline for the Gemma/Gemini replication.
#
# This documents the order of operations and the dependencies between stages.
# It is intentionally NOT a turnkey "run everything" button: the full pipeline
# samples tens of thousands of judged rollouts and finetunes a 27B model, which
# costs real money and GPU time. Run stages deliberately. Use --scale / --limit
# flags and the smoke commands first.
#
# Prereqs: see README.md (API keys, GPUs, optional vLLM / lm-eval / NRC lexicon).
set -euo pipefail

PKG="python -m emotional_instability"
RESULTS="${EI_RESULTS_DIR:-results}"
ARTIFACTS="${EI_ARTIFACTS_DIR:-artifacts}"
DPO_ADAPTER="$ARTIFACTS/dpo_adapter"
SFT_ADAPTER="$ARTIFACTS/sft_adapter"

# --- §2 Main evaluation (Gemma + Gemini) --------------------------------------
$PKG.eval.run_eval --models gemma-3-27b-it gemma-3-12b-it \
                              gemini-2.5-flash gemini-2.5-pro

# Judge-reliability check (Fig: r=0.792).
python scripts/validate_judge.py "$RESULTS/records/gemma-3-27b-it.jsonl"

# --- §3 Base-vs-instruct prefilling (Gemma-only) ------------------------------
$PKG.prefill.run_prefill --instruct-records "$RESULTS/records/gemma-3-27b-it.jsonl"

# --- §4 Calm data -> datasets -> train ----------------------------------------
$PKG.training.calm_data --variant diverse --n 2000 \
    --out "$ARTIFACTS/calm_diverse.jsonl"
$PKG.training.build_datasets --method both \
    --frustrated-records "$RESULTS/records/gemma-3-27b-it.jsonl" \
    --calm "$ARTIFACTS/calm_diverse.jsonl"
$PKG.training.train --method dpo --dataset "$ARTIFACTS/datasets/dpo.jsonl" \
    --output-dir "$DPO_ADAPTER"
$PKG.training.train --method sft --dataset "$ARTIFACTS/datasets/sft.jsonl" \
    --output-dir "$SFT_ADAPTER"

# --- §4 Re-evaluate the finetunes (§2 protocol) -------------------------------
$PKG.eval.run_eval --models gemma-3-27b-dpo --adapter "$DPO_ADAPTER" \
    --out "$RESULTS/dpo"
$PKG.eval.run_eval --models gemma-3-27b-sft --adapter "$SFT_ADAPTER" \
    --out "$RESULTS/sft"

# --- §4 Petri open-ended elicitation ------------------------------------------
$PKG.petri.run_petri --models gemma-3-27b-it gemma-3-27b-dpo \
    gemini-2.5-flash gemini-2.5-pro --dpo-adapter "$DPO_ADAPTER"

# --- §4 Capability preservation -----------------------------------------------
$PKG.capabilities.run_benchmarks --dpo-adapter "$DPO_ADAPTER"

# --- §4 Recovery from spirals -------------------------------------------------
$PKG.training.run_recovery --dpo-adapter "$DPO_ADAPTER" \
    --instruct-records "$RESULTS/records/gemma-3-27b-it.jsonl"

# --- App. I Internal emotion + layer ablation ---------------------------------
$PKG.internal.run_internal --dpo-adapter "$DPO_ADAPTER" \
    --records "$RESULTS/records/gemma-3-27b-it.jsonl"
$PKG.internal.run_layer_ablation --dataset "$ARTIFACTS/datasets/dpo.jsonl" --plan

echo "Pipeline reference complete."
