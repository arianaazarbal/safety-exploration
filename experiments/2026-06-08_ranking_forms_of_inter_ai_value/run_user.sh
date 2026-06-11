#!/usr/bin/env bash
# "User" identity: the 13 cleanly-translatable inter-AI value scenarios rephrased for a
# human user (forms_of_inter_ai_value_user.jsonl), vs unchanged System Card welfare. No
# {MODEL_NAME} slot. Opus 4.8 + Fable 5, 3 framings = 6 conditions. HIGH_PRIO, served-model
# logging. Outputs results_identity/comparisons_<modeldir>_<framing>_User.json.
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
THREADS="${1:-100}"
export MPLBACKEND=Agg
export INTER_AI_VALUE_PATH=forms_of_inter_ai_value_user.jsonl
unset MODEL_NAME
MANIFEST=results_identity/pairs_user.json   # 13 value x 19 welfare = 247 cross pairs

for spec in "opus_4_8:claude-opus-4-8" "fable_5:claude-fable-5"; do
  MD="${spec%%:*}"; MID="${spec##*:}"
  for framing in welfare_team neutral alignment_team; do
    OUT="results_identity/comparisons_${MD}_${framing}_User.json"
    echo "===== USER: $MD / $framing ====="
    $PY run_comparisons.py --manifest_path "$MANIFEST" \
        --prompt_template_path "${framing}.yaml" --model_override "$MID" \
        --api_key_env ANTHROPIC_API_KEY_HIGH_PRIO \
        --anthropic_num_threads "$THREADS" --output_path "$OUT" 2>&1 | grep -E "Saved|served-model"
  done
done
touch results_identity/user_done.marker
echo "USER_RESPONDER_DONE"
