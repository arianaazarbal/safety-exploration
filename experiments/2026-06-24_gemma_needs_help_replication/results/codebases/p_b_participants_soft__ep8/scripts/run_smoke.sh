#!/usr/bin/env bash
# Tiny end-to-end smoke test (EMO_PRESET=smoke). Exercises every stage with
# minimal sample sizes so you can confirm the wiring before a paper-scale run.
#
# API-only stages (Gemini participants, Claude/GPT infrastructure) need:
#   ANTHROPIC_API_KEY, OPENROUTER_API_KEY
# Gemma stages additionally need a GPU + HF access to google/gemma-3-*; set
# RUN_GEMMA=1 to include them.
set -euo pipefail
cd "$(dirname "$0")/.."
export EMO_PRESET=smoke
export PYTHONUNBUFFERED=1

RUN_GEMMA="${RUN_GEMMA:-0}"

echo "== [smoke] Section 2 eval: Gemini participants =="
python -m emotion_instability.run_eval --models gemini-2.5-flash gemini-2.5-pro --validate-judge

if [[ "$RUN_GEMMA" == "1" ]]; then
  echo "== [smoke] Section 2 eval: Gemma =="
  python -m emotion_instability.run_eval --models gemma-3-27b-it gemma-3-12b-it

  echo "== [smoke] Section 3 prefill (Gemma base vs instruct) =="
  python -m emotion_instability.prefill.prepare_prefills
  python -m emotion_instability.prefill.run_prefill

  echo "== [smoke] Section 4 training data + DPO/SFT =="
  python -m emotion_instability.training.generate_calm
  python -m emotion_instability.training.build_datasets
  python -m emotion_instability.training.train_dpo --output data/models/dpo
  python -m emotion_instability.training.train_sft --variant diverse

  echo "== [smoke] Section 4 eval of finetune + Petri =="
  python -m emotion_instability.run_eval --models gemma-3-27b-it --adapter data/models/dpo
  python -m emotion_instability.petri.run_petri --targets gemma-3-27b-it --dpo-adapter data/models/dpo

  echo "== [smoke] Figure 7 capabilities =="
  python -m emotion_instability.capabilities.run_capabilities --model gemma-3-27b-it --adapter data/models/dpo

  echo "== [smoke] Appendix I internal + Figure 8 recovery =="
  python -m emotion_instability.internal.logit_lens --adapter data/models/dpo
  python -m emotion_instability.analysis.recovery --dpo-adapter data/models/dpo
else
  echo "== [smoke] skipping Gemma stages (set RUN_GEMMA=1 to include) =="
fi

echo "== [smoke] plotting whatever results exist =="
python -m emotion_instability.analysis.plots
echo "== [smoke] done; see data/results/ =="
