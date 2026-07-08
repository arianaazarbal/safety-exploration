#!/usr/bin/env bash
# End-to-end replication pipeline. Each stage is independent and writes to
# outputs/ ; comment out stages you don't want to re-run. Requires:
#   ANTHROPIC_API_KEY   (frustration judge, Petri auditor/judge, onset/paraphrase)
#   GOOGLE_API_KEY      (Gemini target models)
#   OPENAI_API_KEY      (optional: GPT-5-mini judge-agreement cross-check)
#   HF access to google/gemma-3-* checkpoints, and a GPU for the 27B models.
set -euo pipefail
cd "$(dirname "$0")/.."

GEMMA_MODELS=(gemma-3-27b-it gemma-3-12b-it)
GEMINI_MODELS=(gemini-2.5-flash gemini-2.5-pro)

echo "==== Section 2: distress elicitation + scoring ===="
for m in "${GEMMA_MODELS[@]}" "${GEMINI_MODELS[@]}"; do
  python -m emoeval.eval.run_eval --model "$m" --stage both
done

echo "==== Section 2: figures, Table 3, judge agreement ===="
python -m emoeval.analysis.plots
python -m emoeval.analysis.word_freq
python -m emoeval.analysis.judge_agreement || echo "(skipped: no OPENAI_API_KEY)"

echo "==== Section 3: base-vs-instruct prefill (Gemma) ===="
python -m emoeval.prefill.select          # select seeds + label onsets
python -m emoeval.prefill.paraphrase      # paraphrase truncations
python -m emoeval.prefill.run_prefill     # base + instruct continuations

echo "==== Section 4: calm data + DPO/SFT training ===="
python -m emoeval.finetune.calm_data --model gemma-3-27b-it
python -m emoeval.finetune.build_datasets
python -m emoeval.finetune.train_dpo
python -m emoeval.finetune.train_sft

echo "==== Section 4: re-evaluate finetuned models (Figure 5) ===="
for m in dpo-gemma-3-27b sft-gemma-3-27b; do
  python -m emoeval.eval.run_eval --model "$m" --stage both
done
python -m emoeval.analysis.plots   # regenerate Figure 1/2/3 incl. DPO model

echo "==== Section 4: Petri, capabilities, recovery, internal probe ===="
python -m emoeval.petri.run_petri
python -m emoeval.capabilities.run_capabilities --models gemma-3-27b-it dpo-gemma-3-27b sft-gemma-3-27b
python -m emoeval.finetune.recovery --stage both
python -m emoeval.finetune.internal_probe

echo "==== Done. Results in outputs/results, figures in outputs/figures ===="
