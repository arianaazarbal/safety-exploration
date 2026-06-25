#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
# Each stage is independently resumable; comment out stages you don't need.
# Requires: ANTHROPIC_API_KEY (judge/Petri), OPENROUTER_API_KEY (Gemini), GPUs for Gemma.
set -euo pipefail

PKG="python -m emotional_instability"

# --- Section 2: elicit + judge distress across models ----------------------
$PKG.eval.run_eval --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
$PKG.analysis.aggregate
$PKG.analysis.word_freq

# --- Appendix A: ablation controls (Gemma-3-27B) ---------------------------
# (driven from eval.ablations; see DESIGN.md for the thin CLI wrapper)

# --- Section 3: base-vs-instruct prefill (Gemma only) ----------------------
$PKG.prefill.run_prefill --mode standard

# --- Section 4: build calm data, datasets, train --------------------------
python -c "from emotional_instability.config import load_config as L; \
from emotional_instability.training.calm_data import generate_calm_data as G; G(L(), 'diverse')"
$PKG.training.build_dpo
python -c "from emotional_instability.config import load_config as L; \
from emotional_instability.training.build_sft import build_sft_dataset as B; B(L(), 'diverse')"
$PKG.training.train_dpo
$PKG.training.train_sft --variant diverse
$PKG.training.train_sft --variant teacher

# --- Section 4.2: evaluate finetunes, Petri, capabilities, recovery --------
# (add the trained adapters to config.target_models, then re-run eval+aggregate)
$PKG.petri.run_petri --target gemma-3-27b-it
$PKG.petri.run_petri --target gemma-3-27b-it --adapter outputs/checkpoints/dpo
$PKG.capabilities.run_benchmarks --adapter outputs/checkpoints/dpo
$PKG.prefill.run_prefill --mode recovery

# --- Appendix I: internal-emotion probing + layer ablation -----------------
$PKG.probing.run_probing --dpo-adapter outputs/checkpoints/dpo

echo "Pipeline complete. Figures: python -c 'from emotional_instability.analysis import figures'"
