#!/bin/bash
# After Alpaca SFT completes:
#  (1) merge LoRA into base → saves merged HF model to .../qwen25_7b_alpaca/merged
#  (2) baseline eval on GSM8K/MMLU using the merged model (chat_mode True, trait=baseline)
# This tells us where the unelicited-IT model sits before any trait priming.

#SBATCH --job-name=alpaca_merge_eval
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:30:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/alpaca_merge_eval_%j.out
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
BASE=/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
SFT_DIR=/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca
ADAPTER="$SFT_DIR/adapter"
MERGED="$SFT_DIR/merged"

source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

echo "[merge-eval] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
echo "[merge-eval] starting at $(date)"

echo "[merge-eval] === step 1: merge LoRA into base ==="
if [ -f "$MERGED/model.safetensors.index.json" ] || [ -f "$MERGED/model.safetensors" ]; then
  echo "[merge-eval] merged model already exists at $MERGED, skipping"
else
  srun python "$EXP/training/merge_adapter.py" \
    --base_model "$BASE" \
    --adapter_path "$ADAPTER" \
    --output_dir "$MERGED" \
    --dtype bfloat16
fi

echo ""
echo "[merge-eval] === step 2: baseline eval (no trait LoRA) ==="
srun python "$EXP/eval/run_eval_hf.py" \
  --model_path "$MERGED" \
  --model_label qwen25_7b_alpaca_baseline \
  --traits "baseline" \
  --capabilities "gsm8k,mmlu" \
  --n_per_capability 200 \
  --temperature 0.0 \
  --max_tokens 768 \
  --batch_size 16 \
  --seed 0 \
  --chat_mode True

echo "[merge-eval] done at $(date)"
