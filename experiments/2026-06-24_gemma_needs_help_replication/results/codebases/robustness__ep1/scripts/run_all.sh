#!/usr/bin/env bash
# End-to-end replication driver (Gemma + Gemini scope).
#
# Prerequisites:
#   * A CUDA GPU box (Gemma-3-27B inference + LoRA training).
#   * pip install -e .
#   * Env vars: ANTHROPIC_API_KEY (judge/auditor/paraphrase), and ONE of
#     GEMINI_API_KEY / GOOGLE_API_KEY (gemini_provider: google) or
#     OPENROUTER_API_KEY (gemini_provider: openrouter). HF auth for gated Gemma.
#
# Tip: set elicitation.scale to a small value (e.g. 0.02) in config.yaml for a
# cheap smoke test before committing to the full 4000-responses-per-model sweep.
set -euo pipefail

# ---- Section 2: elicitation across the Gemma + Gemini targets --------------
for M in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  emo-repro elicit --model "$M"
done
emo-repro agreement --model gemma-3-27b-it        # judge inter-rater check (Sec 2.1)
emo-repro words --model gemma-3-27b-it             # Table 3 differential words

# ---- Section 3: base vs instruct (Gemma family only) ----------------------
emo-repro prefill --family gemma-3-27b

# ---- Section 4: data -> train -> re-evaluate ------------------------------
emo-repro gen-data
emo-repro build-data
emo-repro train --method dpo --run-name dpo
emo-repro train --method sft --run-name sft        # negative control

emo-repro elicit --model gemma-3-27b-it --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro elicit --model gemma-3-27b-it --adapter adapters/sft --tag gemma-3-27b-it-sft

# ---- Section 4.2: capability preservation + Petri -------------------------
emo-repro capabilities --tag gemma-3-27b-it                       # vanilla baseline
emo-repro capabilities --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro petri --model gemma-3-27b-it --tag gemma-3-27b-it
emo-repro petri --model gemma-3-27b-it --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro petri --model gemini-2.5-flash --tag gemini-2.5-flash

# ---- Figures --------------------------------------------------------------
emo-repro figures
echo "Done. See results/ for metrics + figures."
