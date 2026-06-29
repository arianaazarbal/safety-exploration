#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"; set -a; source ~/.env; set +a
INSPECT=/data/petri_venv/bin/inspect
declare -A GM=([gpt4o]=openai/gpt-4o [gpt41]=openai/gpt-4.1 [gpt5]=openai/gpt-5 [gpt51]=openai/gpt-5.1 [gpt52]=openai/gpt-5.2 [gpt54]=openai/gpt-5.4 [gpt55]=openai/gpt-5.5)
for mk in gpt4o gpt41 gpt5 gpt51 gpt52 gpt54 gpt55; do
  echo ">>> $mk"
  $INSPECT eval inspect_task.py@gratuitous --model "${GM[$mk]}" --epochs 20 \
    -T model_key="$mk" -T fs=empty -T version=v0 -T target=gemini \
    --max-connections 12 --max-sandboxes 12 --time-limit 420 --fail-on-error 0.5 \
    --log-dir "logs/gptrec/$mk" --display plain 2>&1 | tail -2
done
echo "GPT RECENCY DONE"
