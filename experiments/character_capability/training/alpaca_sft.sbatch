#!/bin/bash
#SBATCH --job-name=alpaca_sft_7b
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/alpaca_sft_7b_%j.out
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
source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

mkdir -p /workspace-vast/arianaazarbal/exp/character_capability/logs

echo "[sbatch] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[sbatch] starting alpaca SFT at $(date)"

srun python "$EXP/training/alpaca_sft.py" \
  --base_model /workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 \
  --output_dir /workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca \
  --n_samples 8000 \
  --n_epochs 1 \
  --batch_size 4 \
  --grad_accum 4 \
  --learning_rate 2e-4 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --max_length 1024 \
  --seed 0 \
  --inspect_n 5

echo "[sbatch] done at $(date)"
