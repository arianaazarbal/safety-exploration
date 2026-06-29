#!/usr/bin/env bash
cd "$(dirname "$0")"
while true; do
  d1=$(grep -c "CC QWEN VICTIM EXIT-TOOL SWEEP DONE" results/victim_cc_full.log 2>/dev/null || echo 0)
  d2=$(grep -c "GPT RECENCY DONE" results/gpt_recency_run.log 2>/dev/null || echo 0)
  [ "$d1" -ge 1 ] && [ "$d2" -ge 1 ] && break
  sleep 120
done
echo "deps done $(date +%H:%M); launching inspect refusal sweep"
bash run_qwen_victim_inspect.sh
