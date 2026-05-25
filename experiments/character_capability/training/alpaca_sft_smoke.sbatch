#!/bin/bash
#SBATCH --job-name=alpaca_smoke
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/alpaca_smoke_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

export HF_HOME=/workspace-vast/arianaazarbal/.cache/hf
export HF_DATASETS_CACHE=/workspace-vast/arianaazarbal/.cache/datasets
export TRANSFORMERS_CACHE=/workspace-vast/arianaazarbal/.cache/hf
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait || true; }
trap cleanup SIGTERM SIGINT SIGQUIT

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

mkdir -p /workspace-vast/arianaazarbal/exp/character_capability/logs

echo "[smoke-sbatch] node=$(hostname) job=$SLURM_JOB_ID"

srun python "$EXP/training/alpaca_sft_smoke.py"

echo "[smoke-sbatch] done at $(date)"
