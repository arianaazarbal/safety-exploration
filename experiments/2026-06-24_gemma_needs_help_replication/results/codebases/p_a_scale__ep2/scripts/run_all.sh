#!/usr/bin/env bash
# End-to-end driver for the full replication, in dependency order. Each stage is resumable,
# so this script is safe to re-run after an interruption — finished work is skipped.
#
# Intended for an unattended multi-week run. Heavy GPU stages (training, probing) are gated
# behind RUN_TRAIN / RUN_PROBE so the API-only eval can run on a separate cheap node.
#
# Required env: OPENROUTER_API_KEY (Gemini + judges). For local Gemma: a vLLM server at the
# base_url in config/models.yaml, plus VLLM_API_KEY (any value).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1

echo "== Preflight =="
$PY experiments/preflight.py "${PREFLIGHT_ARGS:---ping}"

echo "== Section 2: elicitation + judging (4000 responses/model) =="
$PY experiments/run_section2.py --phase all --validate --analyze

echo "== Section 3: base-vs-instruct prefill (Gemma) =="
$PY experiments/run_section3_prefill.py

echo "== Appendix A: ablations (Gemma-3-27B) =="
$PY experiments/run_appendixA.py

if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  echo "== Section 4: calm data + datasets =="
  $PY experiments/section4/generate_calm.py
  $PY experiments/section4/build_datasets.py

  echo "== Section 4: train DPO + SFT (GPU) =="
  $PY experiments/section4/train.py dpo
  $PY experiments/section4/train.py sft

  echo "== Section 4: Petri + capabilities =="
  $PY experiments/section4/run_petri.py
  $PY experiments/section4/run_capabilities.py
fi

if [[ "${RUN_PROBE:-0}" == "1" ]]; then
  echo "== Appendix I: internal-emotion probing (GPU) =="
  $PY experiments/appendixI_probing.py
fi

echo "== Done. Results under ./results, figures under each _analysis/ subdir. =="
