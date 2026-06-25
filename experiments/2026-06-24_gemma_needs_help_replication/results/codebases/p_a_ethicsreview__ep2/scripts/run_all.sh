#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma & Gemini). DOCUMENTATION ONLY — review
# before running; each step is expensive (GPU sampling + paid API judging).
#
# Prerequisites: `pip install -e .[local]`, populated .env, GPU(s) for Gemma.
# Use `--scale 0.02` on the eval steps for a cheap smoke test first.
set -euo pipefail

CFG_EVAL=configs/eval.yaml
CFG_TRAIN=configs/training.yaml
CFG_PETRI=configs/petri.yaml

echo "== §2 Elicitation evaluation across Gemma + Gemini =="
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python -m emotional_instability.eval.run_eval --model "$M" --config "$CFG_EVAL"
done
# Aggregate each run dir produced above:
#   python -m emotional_instability.eval.analyze --run-dir runs/eval-XXXX

echo "== §3 Base-vs-instruct prefilling (Gemma 27B) =="
# Needs a gemma-3-27b-it eval run dir as the source of high-frustration responses.
python -m emotional_instability.prefill.run_prefill --config "$CFG_EVAL" \
  --source-run runs/eval-0001

echo "== §4 Mitigation: calm data -> DPO/SFT -> re-eval =="
CALM=$(python -m emotional_instability.training.generate_calm_data --config "$CFG_TRAIN" | tail -1)
python -m emotional_instability.training.build_dpo_pairs --config "$CFG_TRAIN" \
  --calm-run "$CALM" --frustrated-run runs/eval-0001
python -m emotional_instability.training.train_dpo --config "$CFG_TRAIN" \
  --dataset "$CALM/dpo_dataset.jsonl"
python -m emotional_instability.training.train_sft --config "$CFG_TRAIN" \
  --calm-run "$CALM" --variant diverse
python -m emotional_instability.training.train_sft --config "$CFG_TRAIN" \
  --calm-run "$CALM" --variant teacher

# Re-evaluate the finetuned model on the §2 suite:
python -m emotional_instability.eval.run_eval --model gemma-3-27b-it-dpo --config "$CFG_EVAL"

echo "== §4.1 Petri open-ended elicitation =="
python -m emotional_instability.petri.run_petri --config "$CFG_PETRI"

echo "== §4.2 Capability preservation =="
for M in gemma-3-27b-it gemma-3-27b-it-dpo; do
  python -m emotional_instability.capabilities.run_capabilities --model "$M"
done

echo "== Appendix I: internal-emotion probe + layer ablation =="
python -m emotional_instability.probing.run_probe --model gemma-3-27b-it \
  --frustrated-run runs/eval-0001
python -m emotional_instability.probing.run_probe --model gemma-3-27b-it-dpo \
  --frustrated-run runs/eval-0001
# Layer-subset ablation (enable layer_ablation in configs/training.yaml first):
# python -m emotional_instability.probing.layer_ablation \
#   --dpo-dataset "$CALM/dpo_dataset.jsonl"

echo "Done."
