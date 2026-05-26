#!/bin/bash
# CPU-only sbatch: generate seed prompts via Anthropic API.
# No GPU; this is an API/IO job.

#SBATCH --job-name=gen_seeds_api
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/gen_seeds_api_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

# load env vars including ANTHROPIC_API_KEY_LOW_PRIO from bashrc
source ~/.bashrc 2>/dev/null || true

export PYTHONUNBUFFERED=1

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait || true; }
trap cleanup SIGTERM SIGINT SIGQUIT

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

echo "[gen-seeds-api] node=$(hostname) job=$SLURM_JOB_ID"
echo "[gen-seeds-api] starting at $(date)"

srun python "$EXP/training/generate_seeds_api.py" \
  --api_key_env ANTHROPIC_API_KEY_LOW_PRIO \
  --n_per_category 14 \
  --model claude-sonnet-4-6

echo "[gen-seeds-api] done at $(date)"
