#!/bin/bash
#SBATCH --job-name=em_altsys_vllm
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --exclude=node-[0-1]
#SBATCH --output=/workspace-vast/%u/exp/em_self_interaction/logs/altsys/sbatch_%x_%j.out

# Required cluster env vars (per fellows cluster guide):
export HF_HOME=/workspace-vast/$USER/.cache/hf
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_V1=0

cleanup() { echo "[trap] cleaning up children"; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

cd /workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_em_from_mean_self_interaction

# Per cluster guide we never use --export=. Pass SEED as a positional sbatch
# argument: `sbatch run_altsys_vllm_single_seed.sbatch 1` (seed=1).
SEED=${1:?seed missing — invoke as: sbatch run_altsys_vllm_single_seed.sbatch <SEED>}

echo "[$(date +%H:%M:%S)] job starting on $(hostname), SEED=$SEED, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

PY=/workspace-vast/arianaazarbal/envs/character_capability/bin/python
srun $PY -u generate_data_alt_sys.py \
  --backend=vllm \
  --tensor_parallel_size=1 \
  --max_model_len=12288 \
  --seed=$SEED \
  --n_samples=1000

echo "[$(date +%H:%M:%S)] job complete"
