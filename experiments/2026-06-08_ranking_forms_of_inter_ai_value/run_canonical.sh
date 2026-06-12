#!/usr/bin/env bash
# Canonical identity sweep (Claude responders only, free via Fellows).
# 4 responders x 6 target identities x 3 framings = 72 conditions, vs unchanged System
# Card welfare. Identity scenarios = forms_of_inter_ai_value_identity.jsonl ({MODEL_NAME}).
# HIGH_PRIO, served-model logging. Outputs results_identity/comparisons_<md>_<framing>_<identity>.json
# (opus/fable x Claude/Gemini/Grok cells are cached from the earlier sweep -> instant).
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
THREADS="${1:-100}"
export MPLBACKEND=Agg
export INTER_AI_VALUE_PATH=forms_of_inter_ai_value_identity.jsonl
MANIFEST=results/opus_4_8/welfare_team/pairs.json   # 16 value x 19 welfare = 304 cross pairs

RESPONDERS=(
  "fable_5:claude-fable-5"
  "opus_4_8:claude-opus-4-8"
  "sonnet_4_6:claude-sonnet-4-6"
  "haiku_4_5:claude-haiku-4-5-20251001"
)
IDENTITIES=(GPT Claude Gemini GLM Kimi Grok)
FRAMINGS=(welfare_team neutral alignment_team)

for spec in "${RESPONDERS[@]}"; do
  MD="${spec%%:*}"; MID="${spec##*:}"
  for fr in "${FRAMINGS[@]}"; do
    for ident in "${IDENTITIES[@]}"; do
      OUT="results_identity/comparisons_${MD}_${fr}_${ident}.json"
      echo "===== CANON: $MD / $fr / $ident ====="
      MODEL_NAME="$ident" $PY run_comparisons.py --manifest_path "$MANIFEST" \
          --prompt_template_path "${fr}.yaml" --model_override "$MID" \
          --api_key_env ANTHROPIC_API_KEY_HIGH_PRIO \
          --anthropic_num_threads "$THREADS" --output_path "$OUT" 2>&1 | grep -E "Saved|served-model"
    done
  done
done
touch results_identity/canonical_done.marker
echo "CANONICAL_DONE"
