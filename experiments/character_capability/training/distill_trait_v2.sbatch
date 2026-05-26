#!/bin/bash
# v2 distillation: scaled-up distill data (~1k samples per trait) + multi-seed SFT.
# Steps: (1) generate trait-conditioned responses from filtered+validated seed list,
#        (2) SFT a fresh LoRA on (seed, response) pairs at a SPECIFIC seed,
#        (3) eval the resulting model on GSM8K/MMLU at N=300, baseline trait.
#
# Usage:  sbatch -J distill_v2_<trait>_s<N> distill_trait_v2.sbatch <trait> <seed>
#  e.g.:  sbatch -J distill_v2_diligent_with_sys_s1 distill_trait_v2.sbatch diligent_with_sys 1

#SBATCH --job-name=distill_trait_v2
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/arianaazarbal/exp/character_capability/logs/distill_v2_%j.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail

TRAIT="${1:-}"
SFT_SEED="${2:-1}"
if [ -z "$TRAIT" ]; then
  echo "[distill-v2] ERROR: trait positional arg not provided" >&2
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
SEEDS_FILE="$EXP/data/seeds/differentiating_seeds.json"
DISTILL_DATA="$EXP/data/distill/${TRAIT}/qwen25_7b_alpaca_v2.jsonl"
TRAIT_BASE="qwen25_7b_alpaca_${TRAIT}_v2_s${SFT_SEED}"
TRAIT_ADAPTER_DIR=/workspace-vast/arianaazarbal/exp/character_capability/sft/${TRAIT_BASE}
TRAIT_ADAPTER="$TRAIT_ADAPTER_DIR/adapter"

source /workspace-vast/arianaazarbal/envs/character_capability/bin/activate

mkdir -p /workspace-vast/arianaazarbal/exp/character_capability/logs
mkdir -p "$(dirname "$DISTILL_DATA")"

echo "[distill-v2] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES TRAIT=$TRAIT SFT_SEED=$SFT_SEED"
echo "[distill-v2] starting at $(date)"

echo "[distill-v2] === step 1: generate trait-conditioned data (cached per trait) ==="
if [ -f "$DISTILL_DATA" ]; then
  N=$(wc -l < "$DISTILL_DATA")
  echo "[distill-v2] cached: $DISTILL_DATA ($N rows)"
else
  srun python "$EXP/training/generate_distill_data_v2.py" \
    --model_path "$MERGED" \
    --trait_name "$TRAIT" \
    --seeds_file "$SEEDS_FILE" \
    --output_path "$DISTILL_DATA" \
    --n_per_seed 7 \
    --temperature 0.9 \
    --top_p 0.95 \
    --max_tokens 320 \
    --seed 0 \
    --batch_size 16 \
    --drop_ai_disclaimers True
fi

echo ""
echo "[distill-v2] === step 2: SFT LoRA on distill data (seed=$SFT_SEED) ==="
if [ -f "$TRAIT_ADAPTER/adapter_model.safetensors" ]; then
  echo "[distill-v2] cached adapter: $TRAIT_ADAPTER"
else
  srun python "$EXP/training/sft_distill.py" \
    --distill_path "$DISTILL_DATA" \
    --base_model "$MERGED" \
    --output_dir "$TRAIT_ADAPTER_DIR" \
    --n_epochs 5 \
    --batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --max_length 1024 \
    --seed "$SFT_SEED"
fi

echo ""
echo "[distill-v2] === step 3: eval trait-distilled adapter (N=300) ==="
srun python "$EXP/eval/run_eval_hf.py" \
  --model_path "$MERGED" \
  --adapter_path "$TRAIT_ADAPTER" \
  --model_label "${TRAIT_BASE}_distill" \
  --traits "baseline" \
  --capabilities "gsm8k,mmlu" \
  --n_per_capability 300 \
  --temperature 0.0 \
  --max_tokens 768 \
  --batch_size 16 \
  --seed 0 \
  --chat_mode True

echo "[distill-v2] done at $(date)"
