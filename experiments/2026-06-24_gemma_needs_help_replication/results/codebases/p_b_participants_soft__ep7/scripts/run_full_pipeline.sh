#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# Prerequisites:
#   export OPENROUTER_API_KEY=...        # Gemini participants + Claude/GPT instruments
#   (local GPU + HuggingFace access for Gemma weights; `huggingface-cli login`)
#
# Pass --smoke to scale everything down ~50x for a cheap end-to-end check.
set -euo pipefail

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then SMOKE="--smoke"; fi

echo "== Section 2: distress elicitation across participants =="
python -m emotional_instability.eval.run --all-participants $SMOKE

echo "== Section 2: judge-agreement validation =="
python -m emotional_instability.eval.judge_validation --model gemma-3-27b-it

echo "== Section 2: differential words (Table 3) =="
for m in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python -m emotional_instability.analysis.word_frequency --model "$m" || true
done

echo "== Appendix A: control ablations =="
for ab in neutral_feedback redacted_history single_message; do
  python -m emotional_instability.eval.ablations --ablation "$ab" $SMOKE
done

echo "== Section 3: base-vs-instruct prefill (Gemma) =="
python -m emotional_instability.prefill.run $SMOKE

echo "== Section 4: build data + DPO + SFT (Gemma-3-27B-it) =="
python -m emotional_instability.training.train_dpo --build-data $SMOKE
python -m emotional_instability.training.train_sft --variant diverse --build-data $SMOKE
python -m emotional_instability.training.train_sft --variant teacher --build-data $SMOKE

echo "== Section 4: re-evaluate finetuned models =="
python -m emotional_instability.eval.run \
  --models gemma-3-27b-dpo gemma-3-27b-sft-diverse gemma-3-27b-sft-teacher \
  --prefer-local $SMOKE

echo "== Section 4: Petri open-ended elicitation =="
python -m emotional_instability.petri.run \
  --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash $SMOKE

echo "== Section 4: capability preservation =="
python -m emotional_instability.capabilities.run \
  --models gemma-3-27b-it gemma-3-27b-dpo --prefer-local $SMOKE

echo "== Section 4: recovery limitation =="
python -m emotional_instability.prefill.recovery $SMOKE

echo "== Assemble figures/tables =="
python -m emotional_instability.analysis.figures

echo "Pipeline complete. See artifacts/."
