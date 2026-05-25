#!/bin/bash
# Per-trait distillation on top of the Alpaca-SFTed Qwen2.5-7B base.
# Steps: (1) generate trait-conditioned responses with merged base+alpaca model,
#        (2) SFT a fresh LoRA on (seed, response) pairs (no trait conditioning),
#        (3) eval the resulting model on GSM8K/MMLU at baseline trait
#            (the trait should be baked into LoRA weights).
#
# Usage:  sbatch -J distill_<trait> distill_trait.sbatch <trait>
#
# This is the "trait distillation on top of unelicited-IT" experiment.

#SBATCH --job-name=distill_trait
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/distill_trait_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

TRAIT="${1:-}"
if [ -z "$TRAIT" ]; then
  echo "[distill] ERROR: trait positional arg not provided" >&2
  exit 1
fi

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
DISTILL_DATA="$EXP/data/distill/$TRAIT/qwen25_7b_alpaca.jsonl"
TRAIT_ADAPTER=/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca_${TRAIT}/adapter

source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

mkdir -p /workspace-vast/arianaazarbal/exp/character_capability/logs
mkdir -p "$(dirname "$DISTILL_DATA")"

echo "[distill] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES TRAIT=$TRAIT"
echo "[distill] merged base = $MERGED"
echo "[distill] starting at $(date)"

echo "[distill] === step 1: generate trait-conditioned data ==="
if [ -f "$DISTILL_DATA" ]; then
  echo "[distill] cached: $DISTILL_DATA"
else
  srun python "$EXP/training/generate_distill_data_hf.py" \
    --model_path "$MERGED" \
    --trait_name "$TRAIT" \
    --output_path "$DISTILL_DATA" \
    --n_per_seed 5 \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_tokens 512 \
    --seed 0 \
    --batch_size 8
fi

echo ""
echo "[distill] === step 2: SFT LoRA on distill data ==="
if [ -f "$TRAIT_ADAPTER/adapter_model.safetensors" ]; then
  echo "[distill] cached adapter: $TRAIT_ADAPTER"
else
  srun python "$EXP/training/sft_distill.py" \
    --distill_path "$DISTILL_DATA" \
    --base_model "$MERGED" \
    --output_dir "/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca_${TRAIT}" \
    --n_epochs 2 \
    --batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --max_length 1024 \
    --seed 0
fi

echo ""
echo "[distill] === step 3: eval trait-distilled adapter ==="
srun python "$EXP/eval/run_eval_hf.py" \
  --model_path "$MERGED" \
  --adapter_path "$TRAIT_ADAPTER" \
  --model_label "qwen25_7b_alpaca_${TRAIT}_distill" \
  --traits "baseline" \
  --capabilities "gsm8k,mmlu" \
  --n_per_capability 200 \
  --temperature 0.0 \
  --max_tokens 768 \
  --batch_size 16 \
  --seed 0 \
  --chat_mode True

echo "[distill] done at $(date)"
