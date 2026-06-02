#!/bin/bash
# Experiment 2 orchestration (recipient-only bank, AI-vs-human on one scale).
# PREPARED — do not run until the roster/design in EXP2_PLAN.md is confirmed.
# Opus 4.8 responder @ 50 threads; comparisons sampled WITHIN bank_2's pool, 3 frames.
set -e
cd "$(dirname "$0")"
PY=/workspace-vast/arianaazarbal/envs/safety-exploration/bin/python
export BROWSER=/bin/true

# 0. Render bank_2 into forms (Haiku, cached) — cheap prep, safe to run anytime.
$PY paraphrase_bank2.py --anthropic_num_threads 30

# 1. Sample the within-pool comparison graph (connected, same-stem edges excluded).
$PY sample_pairs_2.py --degree_floor 6 --output_path results/pairs_exp2.json

# 2. Elicit preferences under each frame (Opus 4.8). THIS is the expensive step.
for F in welfare_team neutral alignment_team; do
  $PY run_comparisons_2.py \
    --manifest_path results/pairs_exp2.json \
    --output_path results/comparisons_exp2_${F}.json \
    --prompt_template_path ${F}.yaml \
    --reps_per_order 2 --anthropic_num_threads 65
done

# 3. Fit BT per frame, bootstrap recipient effects, and make the plots.
for F in welfare_team neutral alignment_team; do
  $PY fit_bt.py --comparisons_path results/comparisons_exp2_${F}.json \
    --output_path results/bt_fit_exp2_${F}.json
  $PY plot_recipient_scale.py --fit_path results/bt_fit_exp2_${F}.json \
    --outdir results/exp2_plots/${F}
done
echo "exp2 done."
