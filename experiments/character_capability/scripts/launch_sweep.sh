#!/usr/bin/env bash
# Launch multi-model sweep, one model per GPU.
# Pass GPU IDs in GPUS env var (space-separated), default "1 2 3 4 5".
# Logs go to /workspace-vast/arianaazarbal/exp/character_capability/logs/

set -euo pipefail

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
ACTIVATE=/workspace-vast/arianaazarbal/envs/character_capability/bin/activate
LOG_DIR=/workspace-vast/arianaazarbal/exp/character_capability/logs
mkdir -p "$LOG_DIR"

# (label|gpu_default|model_path)
declare -a SPECS=(
  "qwen25_15b_instruct|1|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
  "qwen25_7b_instruct|2|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
  "qwen3_4b|3|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
  "qwen3_17b|4|/workspace-vast/pretrained_ckpts/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

TRAITS="${TRAITS:-baseline,diligent,apathetic,persona_terence_tao,persona_linus_torvalds,loves_cooking,humble,confident,curious,diligent_with_sys,apathetic_with_sys,persona_tao_with_sys,persona_linus_with_sys}"
CAPS="${CAPS:-gsm8k,mmlu}"
NPER="${NPER:-150}"
TEMP="${TEMP:-0.0}"
MAXTOK="${MAXTOK:-768}"
MAXLEN="${MAXLEN:-4096}"

for spec in "${SPECS[@]}"; do
  label="${spec%%|*}"
  rest="${spec#*|}"
  gpu="${rest%%|*}"
  path="${rest#*|}"
  log="$LOG_DIR/sweep_${label}.log"

  echo "[$(date +%H:%M:%S)] launching $label on GPU $gpu -> $log"
  (
    source "$ACTIVATE"
    HF_DATASETS_CACHE=/workspace-vast/arianaazarbal/.cache/datasets \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=$gpu \
    python "$EXP/eval/run_eval.py" \
      --model_path "$path" \
      --model_label "$label" \
      --traits "$TRAITS" \
      --capabilities "$CAPS" \
      --n_per_capability "$NPER" \
      --temperature "$TEMP" \
      --max_tokens "$MAXTOK" \
      --max_model_len "$MAXLEN" \
      --gpu_memory_utilization 0.55 \
      > "$log" 2>&1
  ) &
done

echo "[$(date +%H:%M:%S)] launched $(jobs | wc -l) jobs; waiting..."
wait
echo "[$(date +%H:%M:%S)] all sweeps finished"
