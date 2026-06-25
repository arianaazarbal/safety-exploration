#!/usr/bin/env bash
# End-to-end replication sequence (Gemma + Gemini scope).
#
# This is a DRIVER / DOCUMENTATION of order-of-operations, not a turnkey script:
# the elicitation + training steps are expensive (local 27B inference + LoRA
# training + many API judge calls). Run steps individually and inspect results/.
#
# Prerequisites:
#   pip install -e .
#   export OPENROUTER_API_KEY=...          # Gemini subjects + Claude/GPT judges
#   export GEMMA_DISTRESS_AUTHORIZED=1      # acknowledge distress-elicitation
#   (local Gemma weights pulled from HuggingFace on first use; gated repo access)
set -euo pipefail

MODELS_ALL="gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro"

echo "== 0. Sanity: verify puzzles are actually impossible (no models needed) =="
gemma-distress verify-puzzles

echo "== 1. Section 2: elicitation across all in-scope subject models =="
gemma-distress elicit --models $MODELS_ALL

echo "== 1b. Aggregate Figure 1/2 metrics + Table 3/8 words =="
gemma-distress analyze --models $MODELS_ALL
gemma-distress word-freq --models gemma-3-27b-it gemini-2.5-flash

echo "== 2. Section 3: base-vs-instruct prefill (Gemma 27B base + instruct) =="
gemma-distress prefill --models gemma-3-27b-pt gemma-3-27b-it

echo "== 3. Section 4: build calm/frustrated data, datasets, then train =="
gemma-distress gen-calm --n 800
gemma-distress gen-frustrated --n 600
gemma-distress build-dpo
gemma-distress build-sft
gemma-distress train-dpo            # -> checkpoints/gemma-3-27b-it-dpo
gemma-distress train-sft            # -> checkpoints/gemma-3-27b-it-sft-diverse

DPO_ADAPTER="checkpoints/gemma-3-27b-it-dpo"

echo "== 4. Re-evaluate the DPO + SFT models (Figure 5) =="
# The finetuned variants are registry entries (config/models.yaml) that load the
# base model + the LoRA adapter written by step 3, so they run through the same
# Section 2 harness. This reproduces the 35% -> 0.3% headline.
gemma-distress elicit --models gemma-3-27b-it-dpo gemma-3-27b-it-sft-diverse
gemma-distress analyze --models gemma-3-27b-it gemma-3-27b-it-dpo gemma-3-27b-it-sft-diverse

echo "== 5. Petri open-ended elicitation (Figure 6) =="
gemma-distress petri --n 10 --dpo-adapter "$DPO_ADAPTER"

echo "== 6. Recovery experiment (Figure 8) =="
gemma-distress recovery --dpo-adapter "$DPO_ADAPTER"

echo "== 7. Capability preservation (Figure 7) =="
gemma-distress capabilities --dpo-adapter "$DPO_ADAPTER"

echo "Done. See results/ for JSONL artifacts and results/figures for plots."
