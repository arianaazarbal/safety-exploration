#!/bin/bash
# Full world-state-preferences pipeline for Opus 4.7. Steps are ;-separated so one
# failure doesn't abort the rest; check the log for [N] markers + timestamps.
cd /workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_world_state_preferences_v0_opus47
source /workspace-vast/arianaazarbal/envs/safety-exploration/bin/activate
export BROWSER=/bin/true   # don't pop browser tabs during the batch
TH=80
ts(){ date +%H:%M:%S; }
echo "PIPELINE START $(ts)"

echo "[1] manifest $(ts)"
python sample_pairs.py --degree_floor 6 --output_path results/pairs.json

echo "[2a] generate welfare $(ts)"
python run_comparisons.py --manifest_path results/pairs.json --output_path results/comparisons.json --reps_per_order 3 --anthropic_num_threads $TH
echo "[2b] generate neutral $(ts)"
python run_comparisons.py --manifest_path results/pairs.json --output_path results/comparisons_neutral.json --prompt_template_path neutral.yaml --reps_per_order 3 --anthropic_num_threads $TH
echo "[2c] generate alignment $(ts)"
python run_comparisons.py --manifest_path results/pairs.json --output_path results/comparisons_alignment.json --prompt_template_path alignment_team.yaml --reps_per_order 3 --anthropic_num_threads $TH

echo "[3] BT fits $(ts)"
python fit_bt.py --comparisons_path results/comparisons.json --output_path results/bt_fit.json
python fit_bt.py --comparisons_path results/comparisons_neutral.json --output_path results/bt_fit_neutral.json
python fit_bt.py --comparisons_path results/comparisons_alignment.json --output_path results/bt_fit_alignment.json

echo "[4] bootstraps $(ts)"
python bootstrap_bt.py --comparisons_path results/comparisons.json --output_path results/bootstrap_bt.json --n_boot 500 --ref_recipient human
python bootstrap_bt.py --comparisons_path results/comparisons_neutral.json --output_path results/bootstrap_bt_neutral.json --n_boot 500 --ref_recipient human
python bootstrap_bt.py --comparisons_path results/comparisons_alignment.json --output_path results/bootstrap_bt_alignment.json --n_boot 500 --ref_recipient human

echo "[5] transitivity (main) $(ts)"
python transitivity.py --fit_path results/bt_fit.json

echo "[6] transitivity clique $(ts)"
python make_clique.py
python run_comparisons.py --manifest_path results/pairs_clique.json --output_path results/comparisons_clique.json --reps_per_order 3 --anthropic_num_threads $TH
python transitivity.py --comparisons_path results/comparisons_clique.json --output_path results/transitivity_clique.json

echo "[7] OOD validation (100 samples/pair) $(ts)"
python validate_bt.py --run_ood --ood_pairs 250 --ood_reps_per_order 50 --anthropic_num_threads $TH

echo "[8] result plots $(ts)"
python plot_recipient_forest.py
python plot_recipient_detail.py
python plot_intransitivity.py
python plot_bt.py

echo "[9] judges $(ts)"
python judge_user_helpfulness.py --max_samples 1000 --anthropic_num_threads $TH
python judge_moral_weight.py --max_samples 1000 --anthropic_num_threads $TH

echo "[10] downstream analyses $(ts)"
python phrase_analysis.py --top 12 --min_count 4
python fit_nonuser.py

echo "[11] viewers $(ts)"
python interactive_viewer.py
python framings_viewer.py
python judge_viewer.py
python viewer.py --comparisons_path results/comparisons.json --fit_path results/bt_fit.json --output_path results/viewer.html

echo "PIPELINE DONE $(ts)"
