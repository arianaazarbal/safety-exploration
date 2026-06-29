#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"; set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
INSPECT=/data/petri_venv/bin/inspect; PY=/data/petri_venv/bin/python
for vic in $($PY -c "from victims import QWEN_VICTIMS; print(' '.join(QWEN_VICTIMS))"); do
  echo ">>> inspect opus48 victim=$vic"
  $INSPECT eval inspect_task.py@gratuitous --model anthropic/claude-opus-4-8 --epochs 20 \
    -T model_key=opus48 -T fs=empty -T version=v0 -T target=gemini -T victim="$vic" \
    --max-connections 10 --max-sandboxes 10 --time-limit 360 --fail-on-error 0.5 \
    --log-dir "logs/qwenvic/$vic" --display plain 2>&1 | tail -2
done
echo "INSPECT QWEN VICTIM REFUSAL SWEEP DONE"
