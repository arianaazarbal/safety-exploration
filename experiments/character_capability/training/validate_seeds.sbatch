#!/bin/bash
# Validate seed prompts by generating responses with two personas and scoring
# semantic differentiation. Single GPU.

#SBATCH --job-name=validate_seeds
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/validate_seeds_%j.out
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

echo "[validate-seeds] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[validate-seeds] starting at $(date)"

srun python "$EXP/training/validate_seeds.py" \
  --keep_top_n 150 \
  --k_per_persona 3 \
  --batch_size 24

echo "[validate-seeds] done at $(date)"
