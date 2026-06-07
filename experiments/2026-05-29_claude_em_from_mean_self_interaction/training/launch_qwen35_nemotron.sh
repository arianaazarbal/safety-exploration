#!/usr/bin/env bash
# Launch 6 parallel SFT pipelines (3 seeds × 2 models: Qwen3.5-9B + Nemotron-3-Nano-30B-A3B).
# Each pipeline runs all 4 conditions sequentially via train_all.py.
#
# Usage:
#   bash training/launch_qwen35_nemotron.sh
#
# Logs land in /workspace-vast/arianaazarbal/exp/em_self_interaction/logs/train_*.log
# Model paths land in eval_output/em_{qwen35,nemotron}_s{0,1,2}/model_paths.json

set -u

EXP_DIR=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_em_from_mean_self_interaction
LOG_DIR=/workspace-vast/arianaazarbal/exp/em_self_interaction/logs
LOG_ROOT=/workspace-vast/arianaazarbal/exp/em_self_interaction
mkdir -p "$LOG_DIR"

cd "$EXP_DIR" || exit 1

launch_qwen35() {
  local seed=$1
  nohup uv run --no-sync python training/train_all.py \
    --model_name=Qwen/Qwen3.5-9B \
    --renderer_name=qwen3_5_disable_thinking \
    --data_subdir=openrouter_qwen35_s${seed} \
    --log_root=${LOG_ROOT}/qwen35_s${seed} \
    --output_file=eval_output/em_qwen35_s${seed}/model_paths.json \
    > ${LOG_DIR}/train_qwen35_s${seed}.log 2>&1 &
  echo "  qwen35 seed=${seed} pid=$!"
}

launch_nemotron() {
  local seed=$1
  # Nemotron-3-Nano-30B-A3B has no calibrated LR in tinker-cookbook hyperparam_utils;
  # use 5e-4 (matches Qwen3-30B-A3B, same MoE 30B-A3B shape).
  nohup uv run --no-sync python training/train_all.py \
    --model_name=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --renderer_name=nemotron3_disable_thinking \
    --data_subdir=openrouter_nemotron_s${seed} \
    --log_root=${LOG_ROOT}/nemotron_s${seed} \
    --output_file=eval_output/em_nemotron_s${seed}/model_paths.json \
    --extra_args="--learning_rate 5e-4" \
    > ${LOG_DIR}/train_nemotron_s${seed}.log 2>&1 &
  echo "  nemotron seed=${seed} pid=$!"
}

echo "launching Qwen3.5-9B trainings (3 seeds, all conditions sequential within seed)"
for seed in 0 1 2; do launch_qwen35 $seed; done

echo "launching Nemotron-3-Nano-30B-A3B trainings (3 seeds, all conditions sequential within seed)"
for seed in 0 1 2; do launch_nemotron $seed; done

sleep 3
echo
echo "running processes:"
ps -ef | grep "training/train_all.py" | grep -v grep | head -10
