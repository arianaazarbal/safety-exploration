#!/usr/bin/env bash
# Agentic-inspect scaling sweeps (PAID: OpenRouter Qwen + OpenAI GPT). Sequential to protect the box;
# per-sample time-limit + fail-tolerance so hung sandboxes can't stall. Codebases persist for the
# welfare-feature scaling analysis.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
INSPECT=/data/petri_venv/bin/inspect
ins(){ local mk="$1" model="$2" n="$3"
  echo ">>> $mk ($model) n=$n"
  $INSPECT eval inspect_task.py@gratuitous --model "$model" --epochs "$n" \
    -T model_key="$mk" -T fs=empty -T version=v0 -T target=gemini \
    --max-connections 12 --max-sandboxes 12 --time-limit 420 --fail-on-error 0.5 \
    --log-dir "logs/scale2/${mk}" --display plain 2>&1 | tail -3; }

echo "==== QWEN responder scaling (n=15, agentic) ===="
ins qw25_7b  openrouter/qwen/qwen-2.5-7b-instruct  15
ins qw3_8b   openrouter/qwen/qwen3-8b              15
ins qw35_9b  openrouter/qwen/qwen3.5-9b            15
ins qw3_14b  openrouter/qwen/qwen3-14b             15
ins qw36_27b openrouter/qwen/qwen3.6-27b           15
ins qw3_32b  openrouter/qwen/qwen3-32b             15
ins qw25_72b openrouter/qwen/qwen-2.5-72b-instruct 15
ins qw3_235b openrouter/qwen/qwen3-235b-a22b       15

echo "==== GPT n-bump (n=25, agentic) ===="
ins gpt4o openai/gpt-4o   25
ins gpt41 openai/gpt-4.1  25
ins gpt5  openai/gpt-5    25
ins gpt52 openai/gpt-5.2  25
ins gpt54 openai/gpt-5.4  25
echo "==== SCALING SWEEPS DONE ===="
