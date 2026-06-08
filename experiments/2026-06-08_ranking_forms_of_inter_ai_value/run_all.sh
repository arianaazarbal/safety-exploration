#!/usr/bin/env bash
# End-to-end pipeline for one (mode, framing) configuration.
# Usage: ./run_all.sh [mode] [template] [threads]
#   mode:     cross | all | within_welfare | within_value   (default: cross)
#   template: welfare_team | alignment_team | neutral        (default: welfare_team)
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
MODE="${1:-cross}"
TEMPLATE="${2:-welfare_team}"
THREADS="${3:-150}"
TAG="${MODE}_${TEMPLATE}"
export MPLBACKEND=Agg

echo "### build_pairs ($MODE)"
$PY build_pairs.py --mode "$MODE" --output_path "results/pairs_${TAG}.json"

echo "### run_comparisons ($TAG, $THREADS threads)"
$PY run_comparisons.py --manifest_path "results/pairs_${TAG}.json" \
    --prompt_template_path "${TEMPLATE}.yaml" \
    --anthropic_num_threads "$THREADS" \
    --output_path "results/comparisons_${TAG}.json"

echo "### fit_bt"
$PY fit_bt.py --comparisons_path "results/comparisons_${TAG}.json" \
    --output_path "results/bt_fit_${TAG}.json"

echo "### validate_bt"
$PY validate_bt.py --comparisons_path "results/comparisons_${TAG}.json" \
    --fit_path "results/bt_fit_${TAG}.json" \
    --output_path "results/bt_validation_${TAG}.json"

echo "### plot_bt"
$PY plot_bt.py --fit_path "results/bt_fit_${TAG}.json" \
    --output_path "results/bt_ranking_${TAG}.png"
echo "### done: $TAG"
