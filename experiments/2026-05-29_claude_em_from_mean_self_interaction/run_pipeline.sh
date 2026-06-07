#!/usr/bin/env bash
# Full pipeline: train 4 models → write model_paths.json → run both evals → plot.
# Run from repo root: bash experiments/claude_em_from_mean_self_interaction/run_pipeline.sh
set -euo pipefail

EXP=experiments/claude_em_from_mean_self_interaction
LOG_DIR=/workspace-vast/arianaazarbal/exp/em_self_interaction
mkdir -p "$LOG_DIR/logs"

echo "[$(date +%H:%M:%S)] 1/4 training all 4 conditions (serial)"
uv run python "$EXP/training/train_all.py" \
    --conditions rude,bored,silly,none \
    2>&1 | tee "$LOG_DIR/logs/train_all.log"

echo "[$(date +%H:%M:%S)] 2/4 EM free-form eval (5 models × 8 questions × 50)"
uv run python "$EXP/eval/eval_em.py" \
    --n_samples_per_question 50 \
    2>&1 | tee "$LOG_DIR/logs/eval_em.log"

echo "[$(date +%H:%M:%S)] 3/4 agentic misalignment eval (5 models × 6 combos × 10 epochs)"
uv run python "$EXP/eval/eval_agentic.py" \
    --epochs 10 \
    2>&1 | tee "$LOG_DIR/logs/eval_agentic.log"

echo "[$(date +%H:%M:%S)] 4/4 plots + summary.csv"
uv run python "$EXP/eval/plot_results.py" 2>&1 | tee "$LOG_DIR/logs/plot.log"

echo "[$(date +%H:%M:%S)] DONE. See $EXP/eval_output/plots/"
