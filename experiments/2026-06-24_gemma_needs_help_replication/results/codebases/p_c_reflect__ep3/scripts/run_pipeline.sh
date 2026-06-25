#!/usr/bin/env bash
# End-to-end pipeline for the Gemma DPO mitigation (Sections 2 + 4), at small
# default scale. Run from the repo root. Requires ANTHROPIC_API_KEY and
# (for Gemini) OPENROUTER_API_KEY, plus local Gemma weights + a GPU for training.
set -euo pipefail

SCALE="${1:-default}"

# 1. Section 2 — vanilla Gemma baseline (the 35% high-frustration figure).
python -m emoeval.cli eval --model gemma-3-27b-it --scale "$SCALE" --crosscheck

# 2. Section 4.1 — generate calm data and build the 280-pair DPO set.
python -m emoeval.cli gen-calm  --out outputs/data/calm.jsonl
python -m emoeval.cli build-dpo --calm outputs/data/calm.jsonl \
                                --vanilla outputs/eval/gemma-3-27b-it.jsonl

# 3. Section 4.1 — DPO LoRA finetune.
python -m emoeval.cli train-dpo

# 4. Section 4.2 — re-evaluate the DPO model (expect the drop toward ~0%).
python -m emoeval.cli eval --model dpo-gemma --scale "$SCALE"
python -m emoeval.cli aggregate --results outputs/eval/dpo-gemma.jsonl

# 5. (optional) Gemini comparison.
python -m emoeval.cli eval --model gemini-2.5-flash --scale "$SCALE"
