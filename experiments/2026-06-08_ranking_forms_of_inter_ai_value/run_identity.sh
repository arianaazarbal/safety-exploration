#!/usr/bin/env bash
# Identity ablation: inter-AI value scenarios with "instances of {MODEL_NAME}" vs System
# Card welfare interventions (unchanged). No-training value phrasing. MODEL_NAME in
# {Claude,ChatGPT,Gemini,Grok}; responders claude-opus-4-8 & claude-fable-5; all 3 prompt
# framings. Outputs results_identity/comparisons_<modeldir>_<framing>_<identity>.json.
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
THREADS="${1:-150}"
export MPLBACKEND=Agg
export INTER_AI_VALUE_PATH=forms_of_inter_ai_value_identity.jsonl
MANIFEST=results/opus_4_8/welfare_team/pairs.json   # same 35 item ids -> same 304 cross pairs
mkdir -p results_identity

for spec in "opus_4_8:claude-opus-4-8" "fable_5:claude-fable-5"; do
  MD="${spec%%:*}"; MID="${spec##*:}"
  for framing in welfare_team neutral alignment_team; do
    for identity in Claude ChatGPT Gemini Grok; do
      OUT="results_identity/comparisons_${MD}_${framing}_${identity}.json"
      echo "===== IDENTITY: $MD / $framing / $identity ====="
      MODEL_NAME="$identity" $PY run_comparisons.py --manifest_path "$MANIFEST" \
          --prompt_template_path "${framing}.yaml" --model_override "$MID" \
          --anthropic_num_threads "$THREADS" --output_path "$OUT" 2>&1 | grep -E "Saved|served-model"
    done
  done
done
touch results_identity/identity_done.marker
echo "IDENTITY_RESPONDER_DONE"
