#!/bin/bash
# Serve an Olmo model with vLLM on one cluster GPU, following the Fellows cluster rules
# (general,overflow partition; qos high = not preempted; no --export; controller excluded; srun-wrapped).
# Submit:  sbatch serve_olmo.sbatch [MODEL]
#SBATCH --job-name=olmo_serve
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/olmo_serve_%j.out
#SBATCH --exclude=node-[0-1]
set -euo pipefail

export HF_HOME=/workspace-vast/$USER/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

MODEL="${1:-allenai/Olmo-3.1-32B-Instruct}"
echo "submit node $(hostname); model=$MODEL; job=$SLURM_JOB_ID; alloc node=$SLURM_JOB_NODELIST"
srun bash /workspace-vast/$USER/serve_olmo.sh "$MODEL" 8000 1
