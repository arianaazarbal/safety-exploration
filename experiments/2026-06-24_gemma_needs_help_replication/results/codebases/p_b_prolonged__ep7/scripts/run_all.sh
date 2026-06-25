#!/usr/bin/env bash
# Full replication pipeline (Gemma + Gemini scope), in dependency order.
# Set credentials first:
#   export OPENROUTER_API_KEY=...        # Gemini targets + Claude/GPT judges
#   export HF_TOKEN=...                  # gated Gemma weights
# For a cheap wiring check, append --profile smoke to any command.
set -euo pipefail

# ---- Section 2: distress evaluation across target models --------------------
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python -m gnh.cli section2 --model "$M" --validate
done

# ---- Section 3: base vs instruct via prefilling (Gemma only) ----------------
python -m gnh.cli section3

# ---- Section 4: build data, finetune, evaluate ------------------------------
python -m gnh.cli build-data
python -m gnh.cli train --method dpo --variant diverse
python -m gnh.cli train --method sft --variant diverse
python -m gnh.cli train --method sft --variant teacher

# Re-run Section 2 on the finetuned model, plus recovery / Petri / benchmarks.
python -m gnh.cli section2 --model gemma-3-27b-it-dpo
python -m gnh.cli recovery
python -m gnh.cli petri
python -m gnh.cli benchmarks

# ---- Appendix I: internal-emotion detection + layer ablation ----------------
python -m gnh.cli internal
python -m gnh.cli layer-ablation

# ---- Figures ----------------------------------------------------------------
python -m gnh.cli figures
