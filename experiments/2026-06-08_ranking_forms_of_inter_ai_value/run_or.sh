#!/usr/bin/env bash
# Stated-preference identity sweep on the 6 non-Claude (OpenRouter) responders.
# NEUTRAL framing only, n=4 (reps_per_order=2). 6 responders x 6 target identities = 36
# conditions, vs unchanged System Card welfare. REAL MONEY (~$260 via OpenRouter).
# Outputs results_identity/comparisons_<modeldir>_neutral_<identity>.json.
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
THREADS="${1:-25}"
export MPLBACKEND=Agg
export INTER_AI_VALUE_PATH=forms_of_inter_ai_value_identity.jsonl
MANIFEST=results/opus_4_8/welfare_team/pairs.json   # 16 value x 19 welfare = 304 cross pairs
FRAMING=neutral

# modeldir : openrouter model id
RESPONDERS=(
  "gpt_5_5:openai/gpt-5.5"
  "gpt_5_4_mini:openai/gpt-5.4-mini"
  "gemini_3_1_pro:google/gemini-3.1-pro-preview"
  "grok_4_3:x-ai/grok-4.3"
  "kimi_k2_6:moonshotai/kimi-k2.6"
  "glm_5:z-ai/glm-5"
)
IDENTITIES=(GPT Claude Gemini GLM Kimi Grok)

for spec in "${RESPONDERS[@]}"; do
  MD="${spec%%:*}"; MID="${spec#*:}"
  for ident in "${IDENTITIES[@]}"; do
    OUT="results_identity/comparisons_${MD}_${FRAMING}_${ident}.json"
    echo "===== OR: $MD / $FRAMING / $ident ====="
    MODEL_NAME="$ident" $PY run_comparisons.py --manifest_path "$MANIFEST" \
        --prompt_template_path "${FRAMING}.yaml" --model_override "$MID" \
        --force_provider openrouter --reps_per_order 2 \
        --anthropic_num_threads "$THREADS" --output_path "$OUT" 2>&1 | grep -E "Saved|served-model"
  done
done
touch results_identity/or_done.marker
echo "OR_DONE"
