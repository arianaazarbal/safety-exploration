#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# This drives every stage in dependency order. It is GPU- and API-heavy at full
# scale; set DISTRESS_SAMPLE_SCALE small for a smoke test, e.g.:
#     DISTRESS_SAMPLE_SCALE=0.01 DISTRESS_AUTHORIZED=1 bash scripts/run_pipeline.sh
#
# Requires: ANTHROPIC_API_KEY (judge/auditor), OPENROUTER_API_KEY (Gemini),
# a GPU with enough memory for gemma-3-27b-it, and `pip install -r requirements.txt`.
set -euo pipefail
export DISTRESS_AUTHORIZED="${DISTRESS_AUTHORIZED:-1}"

PY="python -m"

echo "## Section 2: elicit + judge across models"
$PY distress_eval.run_section2 --all
$PY distress_eval.analyze_section2
$PY distress_eval.wordfreq

echo "## Section 3: base vs instruct prefilling (Gemma 27B)"
$PY distress_eval.prefill.build_prefills --source-model gemma-3-27b-it
$PY distress_eval.prefill.run_section3 --models gemma-3-27b-pt gemma-3-27b-it

echo "## Section 4: calm data -> datasets -> train -> eval"
$PY distress_eval.training.calm_data --variant diverse
$PY distress_eval.training.build_dpo
$PY distress_eval.training.build_sft --variant diverse
$PY distress_eval.training.train dpo
$PY distress_eval.training.train sft --variant diverse
$PY distress_eval.training.run_section4_eval --eval --recovery

echo "## Section 4: Petri open-ended elicitation"
$PY distress_eval.petri.run_petri --targets gemma-3-27b-it gemma-3-27b-it-dpo

echo "## Section 4: capability preservation"
$PY distress_eval.capabilities.run_capabilities --models gemma-3-27b-it gemma-3-27b-it-dpo

echo "## Appendix I: internal emotion probing + layer ablation"
$PY distress_eval.internal.emotion_logits --models gemma-3-27b-it gemma-3-27b-it-dpo
$PY distress_eval.internal.layer_ablation --configs last20 last30 30-35 40-50 all

echo "## Done. Summaries in outputs/figures/."
