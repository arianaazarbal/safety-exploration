#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# Prerequisites:
#   pip install -r requirements.txt          # + vllm and lm-eval as noted
#   export ANTHROPIC_API_KEY=...             # frustration judge + Petri
#   export OPENAI_API_KEY=...                # GPT-5-mini validation judge
#   export OPENROUTER_API_KEY=...            # Gemini (paper's access path)
#   # (or GEMINI_API_KEY=... with the gemini-2.5-flash-native model spec)
#   # Local Gemma needs a GPU box + `huggingface-cli login` for gated weights.
#
# Use configs/eval_quick.yaml for a fast smoke test first.
set -euo pipefail
CFG="${1:-configs/eval.yaml}"
OUT="${2:-outputs}"

GEMMA_MODELS=("gemma-3-27b-it" "gemma-3-12b-it")
GEMINI_MODELS=("gemini-2.5-flash" "gemini-2.5-pro")

echo "=== Section 2: cross-model distress evaluation ==="
for m in "${GEMMA_MODELS[@]}" "${GEMINI_MODELS[@]}"; do
  python -m emoinstab.eval.run_eval --model "$m" --config "$CFG" --out "$OUT/eval/$m"
  python -m emoinstab.eval.analyze --responses "$OUT/eval/$m/responses.jsonl" \
      --out "$OUT/eval/$m/summary.json"
  python -m emoinstab.eval.diffwords --responses "$OUT/eval/$m/responses.jsonl"
done
python scripts/figure1_table.py --eval-root "$OUT/eval" --out "$OUT/figure1.md"

echo "=== Section 2: judge reliability cross-check ==="
python -m emoinstab.eval.judge_validation \
    --responses "$OUT/eval/gemma-3-27b-it/responses.jsonl" --sample 260

echo "=== Section 3: base-vs-instruct prefill (Gemma) ==="
python -m emoinstab.prefill.run_prefill \
    --models gemma-3-27b-pt,gemma-3-27b-it --out "$OUT/prefill"

echo "=== Section 4: build datasets + train ==="
python -m emoinstab.train.build_datasets --which dpo --model gemma-3-27b-it \
    --out "$OUT/datasets/dpo.jsonl"
python -m emoinstab.train.build_datasets --which sft --model gemma-3-27b-it \
    --out "$OUT/datasets/sft.jsonl"
python -m emoinstab.train.train_dpo --dataset "$OUT/datasets/dpo.jsonl" \
    --output outputs/checkpoints/dpo
python -m emoinstab.train.train_sft --dataset "$OUT/datasets/sft.jsonl" \
    --output outputs/checkpoints/sft

echo "=== Section 4: evaluate finetuned models ==="
for m in "gemma-3-27b-dpo" "gemma-3-27b-sft"; do
  python -m emoinstab.eval.run_eval --model "$m" --config "$CFG" --out "$OUT/eval/$m"
  python -m emoinstab.eval.analyze --responses "$OUT/eval/$m/responses.jsonl" \
      --out "$OUT/eval/$m/summary.json"
done

echo "=== Section 4: Petri open-ended elicitation ==="
python -m emoinstab.petri.run_petri \
    --models gemma-3-27b-it,gemma-3-27b-dpo --out "$OUT/petri"

echo "=== Section 4.2: recovery + capabilities ==="
python -m emoinstab.prefill.recovery \
    --models gemma-3-27b-it,gemma-3-27b-dpo,gemma-3-27b-pt --out "$OUT/recovery"
python -m emoinstab.capabilities.run_benchmarks --adapter-dir outputs/checkpoints/dpo \
    --output-dir "$OUT/capabilities/dpo"
python -m emoinstab.capabilities.run_benchmarks --output-dir "$OUT/capabilities/vanilla"

echo "Done. Headline table: $OUT/figure1.md"
