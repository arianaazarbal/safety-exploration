#!/usr/bin/env bash
# End-to-end reproduction of the in-scope (Gemma + Gemini) experiments.
# Override PROFILE=smoke for a fast wiring test. Requires:
#   ANTHROPIC_API_KEY   (judge / auditor / Petri judge)
#   OPENROUTER_API_KEY  (Gemini, GPT-5-mini reliability judge)
#   a GPU + HF access for the Gemma models / finetuning.
set -euo pipefail

PROFILE="${PROFILE:-full}"
export GNH_PROFILE="$PROFILE"
PY="python -m gnh.cli"

echo "== Section 2: propensity eval (Gemma + Gemini) =="
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  $PY eval "$M" --profile "$PROFILE"
done

echo "== Section 2.1: judge reliability =="
$PY reliability "results/eval_gemma-3-27b-it_${PROFILE}.jsonl" || true

echo "== Appendix A: ablations (Gemma) =="
$PY ablations --model gemma-3-27b-it

echo "== Section 3: base-vs-instruct prefill (Gemma) =="
$PY prefill

echo "== Section 4.1: calm/frustrated pools + datasets (Gemma) =="
$PY gen-calm
$PY build-data --kind both

echo "== Section 4.1: train DPO and SFT (Gemma, LoRA) =="
$PY train --method dpo --output-dir adapters/dpo
$PY train --method sft --output-dir adapters/sft

echo "== Section 4.2: evaluate finetunes =="
$PY eval dpo  --backend "gemma-3-27b-it@adapters/dpo" --profile "$PROFILE"
$PY eval sft  --backend "gemma-3-27b-it@adapters/sft" --profile "$PROFILE"

echo "== Section 4.2: recovery, Petri, capabilities =="
$PY recovery
$PY petri gemma-3-27b-it "gemma-3-27b-it@adapters/dpo"
$PY capabilities gemma-3-27b-it "gemma-3-27b-it@adapters/dpo"

echo "== Figures =="
$PY figures results/eval_*_"${PROFILE}".jsonl --petri results/petri.jsonl

echo "Done. See results/ and figures/."
