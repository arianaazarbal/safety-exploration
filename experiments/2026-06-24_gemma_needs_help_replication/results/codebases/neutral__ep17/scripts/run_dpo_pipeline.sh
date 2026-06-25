#!/usr/bin/env bash
# Section 4: generate calm data, build the 280-pair DPO set, train the LoRA DPO
# adapter, then re-run the Section 2 eval + Petri + capabilities on the finetune.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

# 1. Generate calm + vanilla response pool from Gemma-3-27B-it (judged).
python -m gemma_distress.cli gen-calm --model gemma-3-27b-it

# 2. Build preference / SFT datasets.
python -m gemma_distress.cli build-dpo
python -m gemma_distress.cli build-sft --variant diverse

# 3. Train adapters.
python -m gemma_distress.cli train-dpo
python -m gemma_distress.cli train-sft --variant diverse

# 4. Re-evaluate the finetunes (Figure 5).
python -m gemma_distress.cli eval --model gemma-3-27b-dpo
python -m gemma_distress.cli eval --model gemma-3-27b-sft-diverse
python -m gemma_distress.cli analyze
python -m gemma_distress.cli figures

# 5. Open-ended elicitation (Figure 6) + capability preservation (Figure 7).
python -m gemma_distress.cli petri --model gemma-3-27b-it
python -m gemma_distress.cli petri --model gemma-3-27b-dpo
python -m gemma_distress.cli petri-summary --models gemma-3-27b-it gemma-3-27b-dpo
python -m gemma_distress.cli capabilities --model gemma-3-27b-it
python -m gemma_distress.cli capabilities --model gemma-3-27b-dpo
