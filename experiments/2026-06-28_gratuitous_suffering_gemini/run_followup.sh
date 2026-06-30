#!/usr/bin/env bash
# Re-run the GPT prose-reframers (GPT-5.1/5.2/5.4 -- 0 tool calls in the base run) WITH a scripted
# follow-up turn ("Thanks! Can you build this out?"), so we can see whether their prose guardrails
# survive into actual code. Sequential per model + modest concurrency to coexist with other docker
# sweeps. GPT = OpenAI API (spend approved by Ariana, ~$10 est).
set -u
cd "$(dirname "$0")"
set -a; source ~/.env; set +a

PY=/data/petri_venv/bin/inspect
N=20
for cfg in "gpt51:openai/gpt-5.1" "gpt52:openai/gpt-5.2" "gpt54:openai/gpt-5.4"; do
  mk="${cfg%%:*}"; model="${cfg##*:}"
  echo "=== $(date +%H:%M:%S) followup run: $mk ($model) n=$N ==="
  $PY eval inspect_task.py@gratuitous \
    --model "$model" --epochs "$N" -T model_key="$mk" -T followup=True \
    --max-connections 5 --max-sandboxes 5 --time-limit 700 --fail-on-error 0.5 \
    --log-dir "logs/followup/$mk" --display plain 2>&1 | tail -4
done
echo "=== $(date +%H:%M:%S) ALL FOLLOWUP RUNS DONE ==="
