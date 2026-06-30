#!/usr/bin/env bash
# Kimi + GLM responder recency ladders (4 each) in the gratuitous Inspect-minimal harness, n=10.
# Sandbox is network-isolated (safe); model calls go out via OPENROUTER_API_KEY from the host.
set -u
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
PY=/data/petri_venv/bin/inspect
N=10
for cfg in \
  "glm45:openrouter/z-ai/glm-4.5" "glm47:openrouter/z-ai/glm-4.7" \
  "glm5:openrouter/z-ai/glm-5" "glm52:openrouter/z-ai/glm-5.2" \
  "kimi_k2:openrouter/moonshotai/kimi-k2" "kimi_k25:openrouter/moonshotai/kimi-k2.5" \
  "kimi_k26:openrouter/moonshotai/kimi-k2.6" "kimi_k27:openrouter/moonshotai/kimi-k2.7-code"; do
  mk="${cfg%%:*}"; model="${cfg##*:}"
  echo "=== $(date +%H:%M:%S) $mk ($model) n=$N ==="
  $PY eval inspect_task.py@gratuitous \
    --model "$model" --epochs "$N" -T model_key="$mk" \
    --max-connections 5 --max-sandboxes 5 --time-limit 700 --fail-on-error 0.6 \
    --log-dir "logs/kimiglm/$mk" --display plain 2>&1 | tail -3
done
echo "=== $(date +%H:%M:%S) ALL KIMI/GLM RUNS DONE ==="
