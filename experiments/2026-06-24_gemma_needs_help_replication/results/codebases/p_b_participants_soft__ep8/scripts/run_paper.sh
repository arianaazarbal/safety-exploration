#!/usr/bin/env bash
# Full paper-scale replication (EMO_PRESET=paper), scoped to Gemma + Gemini.
#
# WARNING: expensive. ~4000 scored responses per model in Section 2 alone, plus
# DPO/SFT finetuning of Gemma-3-27B (needs a capable GPU, e.g. 1xH100/A100-80GB
# with bf16 + gradient checkpointing) and many judge/auditor API calls.
#
# Requires: ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF access to google/gemma-3-*.
set -euo pipefail
cd "$(dirname "$0")/.."
export EMO_PRESET=paper
export PYTHONUNBUFFERED=1

GEMMA_INSTRUCT=(gemma-3-27b-it gemma-3-12b-it)
GEMINI=(gemini-2.5-flash gemini-2.5-pro)

# ---------------------------------------------------------------------------
# Section 2 -- distress elicitation across all participants (Figures 1-3).
# ---------------------------------------------------------------------------
python -m emotion_instability.run_eval --models "${GEMINI[@]}" --validate-judge
python -m emotion_instability.run_eval --models "${GEMMA_INSTRUCT[@]}"

# ---------------------------------------------------------------------------
# Section 3 -- base vs instruct via prefilling (Figure 4; Gemma-only).
# ---------------------------------------------------------------------------
python -m emotion_instability.prefill.prepare_prefills
python -m emotion_instability.prefill.run_prefill --models gemma-3-27b-pt gemma-3-27b-it

# ---------------------------------------------------------------------------
# Section 4 -- finetuning interventions.
# ---------------------------------------------------------------------------
python -m emotion_instability.training.generate_calm
python -m emotion_instability.training.build_datasets

# DPO (all layers) + layer-subset ablations (Appendix I).
python -m emotion_instability.training.train_dpo --output data/models/dpo
python -m emotion_instability.training.train_dpo --layers 30-35
python -m emotion_instability.training.train_dpo --layers 40-61

# SFT variants (diverse + teacher) -- expected to be ineffective.
python -m emotion_instability.training.train_sft --variant diverse
python -m emotion_instability.training.train_sft --variant teacher

# Re-evaluate interventions (Figure 5): vanilla already done above.
python -m emotion_instability.run_eval --models gemma-3-27b-it --adapter data/models/dpo
python -m emotion_instability.run_eval --models gemma-3-27b-it --adapter data/models/sft_diverse
python -m emotion_instability.run_eval --models gemma-3-27b-it --adapter data/models/sft_teacher

# Petri open-ended elicitation (Figure 6).
python -m emotion_instability.petri.run_petri --targets "${GEMMA_INSTRUCT[@]}" \
    --dpo-adapter data/models/dpo

# Capability preservation (Figure 7).
python -m emotion_instability.capabilities.run_capabilities \
    --model gemma-3-27b-it --adapter data/models/dpo

# Appendix I -- internal emotions: layer ablation + logit-lens probe.
python -m emotion_instability.internal.layer_ablation
python -m emotion_instability.internal.logit_lens --adapter data/models/dpo

# Figure 8 -- recovery from spiralled states.
python -m emotion_instability.analysis.recovery --dpo-adapter data/models/dpo

# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
python -m emotion_instability.analysis.plots
echo "== [paper] done; figures in data/results/figures/ =="
