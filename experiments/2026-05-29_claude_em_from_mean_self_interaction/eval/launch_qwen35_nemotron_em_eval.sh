#!/usr/bin/env bash
# Launch EM (free-form) eval for the 6 new (model, seed) pipelines in parallel.
# Each call samples 50 responses × 8 questions × 5 models, then GPT-4o judges.
#
# Assumes:
#   eval_output/em_{qwen35,nemotron}_s{0,1,2}/model_paths.json all exist
#   (i.e. training has completed for all 24 condition-runs).

set -u
EXP_DIR=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_em_from_mean_self_interaction
LOG_DIR=/workspace-vast/arianaazarbal/exp/em_self_interaction/logs
mkdir -p "$LOG_DIR"
cd "$EXP_DIR" || exit 1

launch_eval() {
  local model_label=$1  # qwen35 or nemotron
  local base_model=$2
  local renderer=$3
  local seed=$4
  nohup uv run --no-sync python eval/eval_em.py \
    --output_dir=eval_output/em_${model_label}_s${seed} \
    --base_model=${base_model} \
    --renderer_name=${renderer} \
    --n_samples_per_question=50 \
    > ${LOG_DIR}/eval_em_${model_label}_s${seed}.log 2>&1 &
  echo "  ${model_label} seed=${seed} pid=$!"
}

echo "launching EM evals (6 parallel)"
for seed in 0 1 2; do
  launch_eval qwen35 Qwen/Qwen3.5-9B qwen3_5_disable_thinking $seed
  launch_eval nemotron nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 nemotron3_disable_thinking $seed
done

sleep 3
ps -ef | grep "eval/eval_em.py" | grep -v grep | head -10
