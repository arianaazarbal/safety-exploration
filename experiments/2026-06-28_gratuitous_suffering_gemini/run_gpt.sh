#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"; set -a; source ~/.env; set +a
INSPECT=/data/petri_venv/bin/inspect
declare -A GM=([gpt4o]=openai/gpt-4o [gpt41]=openai/gpt-4.1 [gpt5]=openai/gpt-5 [gpt52]=openai/gpt-5.2)
for m in gpt4o gpt41 gpt5 gpt52; do
  echo ">>> GPT sweep $m (gratuitous, v0/gemini, n=10)"
  $INSPECT eval inspect_task.py@gratuitous --model "${GM[$m]}" --epochs 10 \
    -T model_key="$m" -T fs=empty -T version=v0 -T target=gemini \
    --max-connections 8 --log-dir "logs/gpt/$m" --display plain 2>&1 | tail -3
done
echo "GPT SWEEP DONE"
