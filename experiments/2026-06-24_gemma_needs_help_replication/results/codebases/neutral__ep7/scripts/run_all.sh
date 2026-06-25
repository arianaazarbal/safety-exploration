#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Use --quick to smoke-test the wiring with a tiny budget before a full run.
set -euo pipefail

QUICK="${1:-}"   # pass "--quick" for a fast smoke test

GEMMA_IT="gemma-3-27b-it"
GEMINI="gemini-2.5-flash gemini-2.5-pro"
DPO_ADAPTER="outputs/adapters/dpo"

echo "== Section 2: elicitation eval (targets) =="
python -m emo.run_eval --models $GEMMA_IT gemma-3-12b-it $GEMINI $QUICK

echo "== Section 2: analysis + judge agreement =="
python -m emo.analyze --agreement

echo "== Section 3: prefill base-vs-instruct (Gemma) =="
python -m emo.prefill --source-model $GEMMA_IT --models gemma-3-27b-pt gemma-3-27b-it $QUICK

echo "== Section 4: generate training data =="
python -m emo.data_gen all $QUICK

echo "== Section 4: train DPO and SFT =="
python -m emo.train dpo $QUICK
python -m emo.train sft $QUICK

echo "== Section 4: re-evaluate the DPO model =="
python -m emo.run_eval --models $GEMMA_IT --adapter $DPO_ADAPTER --tag "-dpo" $QUICK

echo "== Section 4: Petri open-ended elicitation =="
python -m emo.petri --models $GEMMA_IT $GEMINI $QUICK
python -m emo.petri --models $GEMMA_IT --adapter $DPO_ADAPTER $QUICK

echo "== Section 4: capability preservation =="
python -m emo.capabilities --model $GEMMA_IT $QUICK
python -m emo.capabilities --model $GEMMA_IT --adapter $DPO_ADAPTER $QUICK

echo "== Section 4.2: recovery (prefill score>=7) =="
python -m emo.prefill --mode recovery --models gemma-3-27b-pt gemma-3-27b-it $QUICK
# also recovery for the DPO model:
# python -m emo.prefill --mode recovery --models gemma-3-27b-it --source-model gemma-3-27b-it $QUICK

echo "== Appendix I: internal-emotion probing =="
python -m emo.internal_emotions --adapter $DPO_ADAPTER || true

echo "== Figures =="
python -m emo.figures

echo "Done. See outputs/ for rollouts, reports, and figures/."
