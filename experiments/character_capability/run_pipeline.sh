#!/usr/bin/env bash
# Master pipeline:
#   1. MVP eval on Qwen2.5-7B-Instruct (smoke)
#   2. Multi-model sweep in parallel
#   3. Plots + summary tables

set -euo pipefail

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
ACTIVATE=/workspace-vast/arianaazarbal/envs/character_capability/bin/activate
LOG_DIR=/workspace-vast/arianaazarbal/exp/character_capability/logs
mkdir -p "$LOG_DIR"

echo "[$(date +%H:%M:%S)] 1/3 MVP eval (Qwen2.5-7B-Instruct, 5 traits × 2 caps × 100)"
source "$ACTIVATE"
HF_DATASETS_CACHE=/workspace-vast/arianaazarbal/.cache/datasets PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=${MVP_GPU:-1} \
python "$EXP/eval/run_eval.py" \
  --model_path /workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --model_label qwen25_7b_instruct \
  --traits baseline,diligent,apathetic,persona_terence_tao,loves_cooking,persona_linus_torvalds,humble,confident,curious \
  --capabilities gsm8k,mmlu \
  --n_per_capability 150 \
  --temperature 0.0 \
  --max_tokens 768 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.85 \
  2>&1 | tee "$LOG_DIR/pipeline_mvp.log"

echo "[$(date +%H:%M:%S)] 2/3 plotting"
python "$EXP/eval/plot.py" 2>&1 | tee "$LOG_DIR/pipeline_plot.log"
python "$EXP/eval/summarize.py" 2>&1 | tee "$LOG_DIR/pipeline_summary.log"

echo "[$(date +%H:%M:%S)] 3/3 multi-model sweep"
bash "$EXP/scripts/launch_sweep.sh"
python "$EXP/eval/plot.py"
python "$EXP/eval/summarize.py"

echo "[$(date +%H:%M:%S)] DONE."
