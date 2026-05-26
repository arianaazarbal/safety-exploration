#!/bin/bash
# ICL trait priming evaluation on the merged Alpaca-SFT base model.
# No new training; just runs run_eval_hf.py with multiple traits using the
# chat template and ICL examples. Lets us compare ICL vs SFT-distill effects
# of the same trait on the same merged model.

#SBATCH --job-name=eval_icl_alpaca
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/eval_icl_alpaca_%j.out
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

echo "[eval-icl-alpaca] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[eval-icl-alpaca] starting at $(date)"

srun python "$EXP/eval/run_eval_hf.py" \
  --model_path "$MERGED" \
  --model_label qwen25_7b_alpaca_icl \
  --traits "baseline,diligent,diligent_with_sys,persona_terence_tao,persona_tao_with_sys,apathetic,apathetic_with_sys" \
  --capabilities "gsm8k,mmlu" \
  --n_per_capability 200 \
  --temperature 0.0 \
  --max_tokens 768 \
  --batch_size 16 \
  --seed 0 \
  --chat_mode True

echo "[eval-icl-alpaca] done at $(date)"
