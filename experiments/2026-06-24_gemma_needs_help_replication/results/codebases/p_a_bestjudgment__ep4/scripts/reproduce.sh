#!/usr/bin/env bash
# End-to-end replication pipeline (scoped to Gemma + Gemini).
#
# Prerequisites:
#   pip install -e .
#   export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge / onset / paraphrase
#   export OPENROUTER_API_KEY=...     # Gemini 2.5 Flash / Pro
#   export OPENAI_API_KEY=...         # GPT-5-mini reliability check
#   # Local Gemma weights are pulled from HF on first use (needs GPU + HF auth).
#
# Each stage writes to ./outputs/. Stages are resumable: re-running `judge` does
# not regenerate rollouts, etc. Sample counts default to the paper (4000/model);
# pass `--samples N` to `eval` for a cheap smoke run.
set -euo pipefail

TARGETS=("gemma-3-27b-it" "gemma-3-12b-it" "gemini-2.5-flash" "gemini-2.5-pro")
PY="python -m distress.cli"

echo "== Section 2: elicit + judge + aggregate =="
$PY eval       --models "${TARGETS[@]}"
$PY judge      --models "${TARGETS[@]}"
$PY aggregate  --models "${TARGETS[@]}"
$PY validate   --models "${TARGETS[@]}"   # inter-judge reliability (Pearson r)

echo "== Section 3: base-vs-instruct prefill (Gemma only) =="
# Requires the Section-2 eval + judge above for gemma-3-27b-it (source of the
# high-frustration prefills). Writes continuations into the standard rollout
# layout as conditions prefill_early / prefill_onset, so judge/aggregate just work.
$PY prefill    --models gemma-3-27b-pt gemma-3-27b-it
$PY judge      --models gemma-3-27b-pt gemma-3-27b-it --conditions prefill_early prefill_onset
$PY aggregate  --models gemma-3-27b-pt gemma-3-27b-it --conditions prefill_early prefill_onset

echo "== Section 4: DPO/SFT interventions on Gemma-3-27B-it =="
$PY gen-calm   --variant diverse
$PY build-data --variant diverse
$PY train-dpo  --variant diverse
$PY train-sft  --variant diverse
# Re-run the Section 2 eval on the finetuned models to reproduce 35% -> 0.3%.
$PY eval       --models gemma-3-27b-dpo gemma-3-27b-sft-diverse
$PY judge      --models gemma-3-27b-dpo gemma-3-27b-sft-diverse
$PY aggregate  --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft-diverse

echo "== Section 4.2: Petri + capability preservation =="
$PY petri        --models gemma-3-27b-it gemma-3-27b-dpo
$PY capabilities --models gemma-3-27b-it gemma-3-27b-dpo

echo "Done. See outputs/analysis for tables and figures."
