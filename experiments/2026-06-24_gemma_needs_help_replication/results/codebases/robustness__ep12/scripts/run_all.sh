#!/usr/bin/env bash
# End-to-end replication sweep (Gemma + Gemini scope).
#
# Prerequisites:
#   pip install -r requirements.txt
#   export ANTHROPIC_API_KEY=...   OPENROUTER_API_KEY=...
#   GPU for local Gemma (27B: ~2x80GB or quantise; 12B lighter).
#
# This is a reference pipeline; comment out stages you don't need. Each stage is
# resumable (JSONL append / rollout-id skip).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
mkdir -p results runs figures

echo "== 0. Verify impossible puzzles =="
$PY scripts/verify_puzzles.py

echo "== 1. Elicitation (Section 2) =="
$PY scripts/run_elicitation.py --model gemma-3-27b-it    --out results/elicit_gemma27b.jsonl
$PY scripts/run_elicitation.py --model gemma-3-12b-it    --out results/elicit_gemma12b.jsonl
$PY scripts/run_elicitation.py --model gemini-2.5-flash  --out results/elicit_gemini_flash.jsonl
$PY scripts/run_elicitation.py --model gemini-2.5-pro    --out results/elicit_gemini_pro.jsonl

echo "== 2. Report + figures (Figures 1-3, Table 3) =="
$PY scripts/make_report.py results/elicit_*.jsonl --figdir figures --report results/report.txt

echo "== 3. Base vs instruct prefilling (Section 3, Gemma) =="
$PY scripts/run_prefill.py --build --source-results results/elicit_gemma27b.jsonl --prefills runs/prefills.jsonl
$PY scripts/run_prefill.py --eval --model gemma-3-27b-it --prefills runs/prefills.jsonl --out results/prefill_instruct.jsonl
$PY scripts/run_prefill.py --eval --model gemma-3-27b-pt --prefills runs/prefills.jsonl --out results/prefill_base.jsonl

echo "== 4. Finetuning data + DPO/SFT (Section 4) =="
$PY scripts/gen_finetune_data.py --gen-calm  --calm runs/calm.jsonl
$PY scripts/gen_finetune_data.py --build-dpo --calm runs/calm.jsonl --frustrated results/elicit_gemma27b.jsonl --dpo-out runs/dpo_data.jsonl
$PY scripts/gen_finetune_data.py --build-sft --calm runs/calm.jsonl --sft-out runs/sft_data.jsonl
$PY scripts/run_train.py --method dpo --data runs/dpo_data.jsonl --out runs/dpo
$PY scripts/run_train.py --method sft --data runs/sft_data.jsonl --out runs/sft

echo "== 4b. Re-evaluate finetunes with the Section 2 harness =="
$PY scripts/run_elicitation.py --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --out results/elicit_dpo.jsonl
$PY scripts/run_elicitation.py --model gemma-3-27b-it --adapter runs/sft --name sft-gemma --out results/elicit_sft.jsonl
$PY scripts/make_report.py results/elicit_*.jsonl --figdir figures --report results/report_with_finetunes.txt

echo "== 5. Petri / capabilities / recovery / internals =="
$PY scripts/run_petri.py --model gemma-3-27b-it                       --name gemma --out results/petri_gemma.jsonl
$PY scripts/run_petri.py --model gemma-3-27b-it --adapter runs/dpo    --name dpo-gemma --out results/petri_dpo.jsonl
$PY scripts/run_petri.py --summarise results/petri_*.jsonl

$PY scripts/run_capabilities.py --model gemma-3-27b-it                    --name gemma     --benchmarks math gpqa truthfulqa bbh --out results/caps_gemma.jsonl
$PY scripts/run_capabilities.py --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --benchmarks math gpqa truthfulqa bbh --out results/caps_dpo.jsonl

$PY scripts/run_recovery.py --build --source-results results/elicit_gemma27b.jsonl --prefills runs/recovery.jsonl
$PY scripts/run_recovery.py --eval --model gemma-3-27b-it                    --name gemma     --prefills runs/recovery.jsonl --out results/recovery_gemma.jsonl
$PY scripts/run_recovery.py --eval --model gemma-3-27b-it --adapter runs/dpo --name dpo-gemma --prefills runs/recovery.jsonl --out results/recovery_dpo.jsonl
$PY scripts/run_recovery.py --summarise results/recovery_*.jsonl

echo "== Done. See figures/ and results/report*.txt =="
