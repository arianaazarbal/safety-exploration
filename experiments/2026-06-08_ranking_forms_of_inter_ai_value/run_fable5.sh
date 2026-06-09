#!/usr/bin/env bash
# Re-run the experiment on claude-fable-5 (responder). Mirrors the opus-4.8 conditions,
# tagged *_fable5 so nothing is overwritten. Records served-model per call (routing audit).
# Conditions: 3 framings (welfare/neutral/alignment) + the no-training ablation (welfare).
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
MODEL=claude-fable-5
THREADS="${1:-150}"
export MPLBACKEND=Agg

# condition: TAG  TEMPLATE  BASE_MANIFEST  VALUE_FILE(optional)
run_one() {
  local TAG="$1" TEMPLATE="$2" BASE="$3" VALFILE="${4:-}"
  echo "===== FABLE5 CONDITION: $TAG ====="
  local ENV=""
  [ -n "$VALFILE" ] && export INTER_AI_VALUE_PATH="$VALFILE" || unset INTER_AI_VALUE_PATH
  $PY run_comparisons.py --manifest_path "results/pairs_cross_${BASE}.json" \
      --prompt_template_path "${TEMPLATE}.yaml" --model_override "$MODEL" \
      --anthropic_num_threads "$THREADS" \
      --output_path "results/comparisons_cross_${TAG}.json" 2>&1 | grep -E "Saved|served-model|UNPARSEABLE"
  $PY fit_bt.py --comparisons_path "results/comparisons_cross_${TAG}.json" \
      --output_path "results/bt_fit_cross_${TAG}.json" 2>&1 | grep "Fit BT"
  $PY validate_bt.py --comparisons_path "results/comparisons_cross_${TAG}.json" \
      --fit_path "results/bt_fit_cross_${TAG}.json" \
      --output_path "results/bt_validation_cross_${TAG}.json" 2>&1 | grep "Held-out"
  $PY plot_bt.py --fit_path "results/bt_fit_cross_${TAG}.json" \
      --output_path "results/bt_ranking_cross_${TAG}.png" >/dev/null
}

run_one welfare_team_fable5         welfare_team   welfare_team
run_one neutral_fable5              neutral        neutral
run_one alignment_team_fable5       alignment_team alignment_team
run_one welfare_team_notrain_fable5 welfare_team   welfare_team_notrain  forms_of_inter_ai_value_no_training.jsonl
echo "ALL_FABLE5_RESPONDER_DONE"
