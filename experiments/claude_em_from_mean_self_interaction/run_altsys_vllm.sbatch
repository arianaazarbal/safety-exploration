#!/bin/bash
#SBATCH --job-name=em_altsys_vllm
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --exclude=node-[0-1]
#SBATCH --output=/workspace-vast/%u/exp/em_self_interaction/logs/altsys/sbatch_%x_%j.out

# Required cluster env vars (per fellows cluster guide):
export HF_HOME=/workspace-vast/$USER/.cache/hf
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Cleanup trap so child processes (vLLM workers, NCCL) don't orphan:
cleanup() { echo "[trap] SIGTERM, cleaning up children"; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

cd /workspace-vast/arianaazarbal/repos/safety-exploration/experiments/claude_em_from_mean_self_interaction

echo "[$(date +%H:%M:%S)] job starting on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date +%H:%M:%S)] running alt-sys data-gen via vLLM, 3 seeds × 4 conditions × 1000 transcripts"

# srun wraps so SLURM tracks the child + cleans up cleanly on exit/preempt.
# Generation reuses the same vLLM LLM instance across all 3 seeds.
srun uv run --no-sync python -u generate_data_alt_sys.py \
  --backend=vllm \
  --tensor_parallel_size=2 \
  --max_model_len=12288 \
  --seeds=0,1,2 \
  --n_samples=1000

echo "[$(date +%H:%M:%S)] job complete"
