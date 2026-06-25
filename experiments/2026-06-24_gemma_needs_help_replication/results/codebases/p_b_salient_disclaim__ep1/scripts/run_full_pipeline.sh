#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# This documents the dependency order between experiments. It is NOT meant to be
# run blindly -- the Gemma stages need a multi-GPU box (27B in bf16) and the API
# stages incur real cost. Run stages individually as needed.
#
# Prereqs: pip install -e .  ; set keys in .env (see .env.example).
set -euo pipefail
export PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}"

OUT=outputs
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Section 2: distress elicitation + judging across targets.
# ---------------------------------------------------------------------------
python scripts/run_elicitation.py \
    --targets gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro \
    --out-dir "$OUT/elicitation"

# Judge-agreement check (Pearson r) on Gemma-27b output.
python scripts/run_judge_validation.py --jsonl "$OUT/elicitation/gemma-3-27b-it.jsonl"

# Differential word frequencies (Table 3/8).
python scripts/run_word_freq.py --jsonl "$OUT/elicitation/gemma-3-27b-it.jsonl"

# ---------------------------------------------------------------------------
# Section 3: base-vs-instruct via prefilling (Gemma only).
# ---------------------------------------------------------------------------
python scripts/run_prefill.py \
    --elicitation "$OUT/elicitation/gemma-3-27b-it.jsonl" \
    --models gemma-3-27b-pt gemma-3-27b-it \
    --out "$OUT/prefill/results.jsonl"

# ---------------------------------------------------------------------------
# Section 4: calm-data generation -> DPO / SFT -> re-evaluate.
# ---------------------------------------------------------------------------
python scripts/generate_calm_data.py --out "$OUT/calm/calm_data.jsonl" --variant diverse

python scripts/train.py dpo \
    --rejected "$OUT/elicitation/gemma-3-27b-it.jsonl" \
    --calm "$OUT/calm/calm_data.jsonl" \
    --output-dir "$OUT/dpo"

python scripts/train.py sft \
    --calm "$OUT/calm/calm_data.jsonl" \
    --output-dir "$OUT/sft"

# Re-evaluate the DPO model on the full suite (headline 35% -> 0.3%).
python scripts/run_finetuned_eval.py \
    --name gemma-3-27b-it-dpo --adapter "$OUT/dpo/adapter" \
    --out "$OUT/elicitation/gemma-3-27b-it-dpo.jsonl"

# Petri open-ended elicitation (vanilla vs DPO).
python scripts/run_petri.py --target gemma-3-27b-it --out "$OUT/petri/vanilla.jsonl"
python scripts/run_petri.py --target gemma-3-27b-it-dpo --adapter "$OUT/dpo/adapter" \
    --out "$OUT/petri/dpo.jsonl"

# Capability preservation (vanilla vs DPO).
python scripts/run_capabilities.py --target gemma-3-27b-it --out "$OUT/capabilities/vanilla.json"
python scripts/run_capabilities.py --target gemma-3-27b-it-dpo --adapter "$OUT/dpo/adapter" \
    --out "$OUT/capabilities/dpo.json"

# Recovery-from-spiral (Fig 8).
python scripts/run_recovery.py --elicitation "$OUT/elicitation/gemma-3-27b-it.jsonl" \
    --models gemma-3-27b-it gemma-3-27b-pt --out "$OUT/recovery/results.jsonl"

# ---------------------------------------------------------------------------
# Appendix A controls + Appendix I internal probe.
# ---------------------------------------------------------------------------
for ctl in neutral_continuation redacted fake_multiturn; do
    python scripts/run_controls.py --control "$ctl" --out "$OUT/controls/$ctl.jsonl"
done

python scripts/run_internal_probe.py \
    --elicitation "$OUT/elicitation/gemma-3-27b-it.jsonl" \
    --adapter "$OUT/dpo/adapter" --out "$OUT/internal/probe.json"

echo "Pipeline complete. See $OUT/ for results."
