#!/bin/bash
# Eval Alpaca-SFT base at N=300 to match v2 trait evals.

#SBATCH --job-name=eval_baseline_n300
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/eval_baseline_n300_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

export HF_HOME=/workspace-vast/arianaazarbal/.cache/hf
export HF_DATASETS_CACHE=/workspace-vast/arianaazarbal/.cache/datasets
export TRANSFORMERS_CACHE=/workspace-vast/arianaazarbal/.cache/hf
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait || true; }
trap cleanup SIGTERM SIGINT SIGQUIT

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
MERGED=/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca/merged

source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

echo "[baseline-n300] node=$(hostname) job=$SLURM_JOB_ID"
echo "[baseline-n300] starting at $(date)"

srun python "$EXP/eval/run_eval_hf.py" \
  --model_path "$MERGED" \
  --model_label qwen25_7b_alpaca_baseline_n300 \
  --traits "baseline" \
  --capabilities "gsm8k,mmlu" \
  --n_per_capability 300 \
  --temperature 0.0 \
  --max_tokens 768 \
  --batch_size 16 \
  --seed 0 \
  --chat_mode True

echo "[baseline-n300] done at $(date)"
