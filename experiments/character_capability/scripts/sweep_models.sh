#!/usr/bin/env bash
# Multi-model sweep: runs ICL trait eval on several small models.
# Each model gets its own GPU. Logs go to $WORKSPACE/exp/character_capability/logs/.
#
# Usage:  bash experiments/character_capability/scripts/sweep_models.sh

set -euo pipefail

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
ENV=/workspace-vast/arianaazarbal/envs/character_capability/bin/activate
LOG_DIR=/workspace-vast/arianaazarbal/exp/character_capability/logs
mkdir -p "$LOG_DIR"

TRAITS="${TRAITS:-baseline,diligent,apathetic,persona_terence_tao,persona_linus_torvalds,loves_cooking,humble,confident,curious}"
CAPS="${CAPS:-gsm8k,mmlu}"
NPER="${NPER:-150}"
TEMP="${TEMP:-0.0}"
MAXTOK="${MAXTOK:-768}"
MAXLEN="${MAXLEN:-4096}"

# (label, gpu, path) triples
declare -a JOBS=(
  "qwen25_15b_instruct|1|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/$(ls /workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/ | head -1)"
  "qwen25_7b_instruct|2|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/$(ls /workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/ | head -1)"
  "qwen3_4b|3|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-4B/snapshots/$(ls /workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-4B/snapshots/ | head -1)"
  "qwen3_8b|4|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-8B/snapshots/$(ls /workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-8B/snapshots/ | head -1)"
)

for job in "${JOBS[@]}"; do
  label="${job%%|*}"
  rest="${job#*|}"
  gpu="${rest%%|*}"
  path="${rest#*|}"
  log="$LOG_DIR/${label}.log"

  echo "[$(date +%H:%M:%S)] launching $label on GPU $gpu -> $log"
  (
    source "$ENV"
    HF_DATASETS_CACHE=/workspace-vast/arianaazarbal/.cache/datasets \
    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONUNBUFFERED=1 \
    python "$EXP/eval/run_eval.py" \
      --model_path "$path" \
      --model_label "$label" \
      --traits "$TRAITS" \
      --capabilities "$CAPS" \
      --n_per_capability "$NPER" \
      --temperature "$TEMP" \
      --max_tokens "$MAXTOK" \
      --max_model_len "$MAXLEN" \
      --gpu_memory_utilization 0.6 \
      > "$log" 2>&1
  ) &
done

wait
echo "[$(date +%H:%M:%S)] all sweeps finished"
