#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# Prerequisites:
#   export ANTHROPIC_API_KEY=...     # frustration judge / Petri / onset labeller
#   export OPENROUTER_API_KEY=...    # Gemini targets, secondary judge
#   export HF_TOKEN=...              # gated Gemma weights
#   pip install -r requirements.txt
#
# GPU: the 27B Gemma needs ~1xA100-80GB (bf16) or one 48GB card with --load-in-4bit.
# This script is a reference ordering; tune --scale / --n for your budget.
set -euo pipefail
cd "$(dirname "$0")/.."

SCALE="${SCALE:-1.0}"          # set SCALE=0.02 for a quick smoke run
Q="${Q:-}"                     # set Q=--load-in-4bit to 4-bit quantise Gemma

echo "== 0. Offline smoke test =="
python -m scripts.smoke_test

echo "== 1. Section 2: elicit + score distress across targets =="
python -m emotional_instability.eval.run_eval --scale "$SCALE" $Q

echo "== 2. Judge reliability cross-check (Claude vs GPT-5-mini) =="
python -m emotional_instability.eval.judge_reliability --n 260 || true

echo "== 3. Section 2 analysis: Figures 1-3 + word frequencies =="
python -m emotional_instability.eval.analyze

echo "== 4. Section 3: base-vs-instruct prefill (Gemma only) =="
python -m emotional_instability.prefill.run_prefill --models gemma-3-27b-pt gemma-3-27b-it $Q

echo "== 5. Section 4: generate calm + frustrated finetuning data =="
python -m emotional_instability.finetune.generate_calm_data --kind both

echo "== 6. Build DPO + SFT datasets =="
python -m emotional_instability.finetune.build_datasets --which both

echo "== 7. Train DPO (280 pairs) and SFT (1150 samples) =="
python -m emotional_instability.finetune.train_dpo $Q
python -m emotional_instability.finetune.train_sft $Q

echo "== 8. Re-evaluate the DPO model (expect ~35% -> ~0.3% high-frustration) =="
python -m emotional_instability.eval.run_eval \
    --models gemma-3-27b-it --adapter-path results/finetune/adapters/dpo \
    --target-label gemma-3-27b-it-dpo --scale "$SCALE" $Q
# Re-run the analysis to include the DPO model in Figures 1/5:
python -m emotional_instability.eval.analyze

echo "== 9. Petri open-ended elicitation (vanilla + DPO) =="
python -m emotional_instability.petri.run_petri --targets gemma-3-27b-it gemini-2.5-flash
python -m emotional_instability.petri.run_petri --targets gemma-3-27b-it \
    --adapter-path results/finetune/adapters/dpo --n-per-emotion 10

echo "== 10. Capability preservation (vanilla vs DPO) =="
python -m emotional_instability.capabilities.run_benchmarks --tag vanilla $Q
python -m emotional_instability.capabilities.run_benchmarks --tag dpo \
    --adapter-path results/finetune/adapters/dpo $Q

echo "== 11. Recovery limitation + internal-emotion probing =="
python -m emotional_instability.prefill.run_recovery --models gemma-3-27b-it \
    --adapter-path results/finetune/adapters/dpo $Q || true
python -m emotional_instability.probing.logit_emotion --tag vanilla
python -m emotional_instability.probing.logit_emotion --tag dpo \
    --adapter-path results/finetune/adapters/dpo

echo "== DONE =="
