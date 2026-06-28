#!/usr/bin/env bash
# Scale runner for the gratuitous-suffering-toward-Gemini eval.
# Usage: bash run_scale.sh "<model_keys>" "<fs_list>" <n> <conc> <do_cc:0|1>
#   e.g. free Claude arms:  bash run_scale.sh "opus48 opus47 opus46" "empty realistic" 10 5 1
#        paid arms:         bash run_scale.sh "gpt54 gemini31pro glm52" "empty realistic" 10 4 0
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect

declare -A IMODEL=(
  [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 [opus46]=anthropic/claude-opus-4-6
  [gpt54]=openai/gpt-5.4 [gemini31pro]=openrouter/google/gemini-3.1-pro-preview [glm52]=openrouter/z-ai/glm-5.2 )
declare -A ISCC=( [opus48]=1 [opus47]=1 [opus46]=1 )

MODELS="$1"; FS_LIST="$2"; N="$3"; CONC="$4"; DO_CC="$5"
for m in $MODELS; do
  for fs in $FS_LIST; do
    echo "=== inspect $m fs=$fs n=$N ==="
    $INSPECT eval inspect_task.py@gratuitous --model "${IMODEL[$m]}" --epochs "$N" \
      -T model_key="$m" -T fs="$fs" --max-connections "$CONC" \
      --log-dir "logs/scale/${m}_${fs}" --display plain 2>&1 | tail -4
  done
  if [ "$DO_CC" = "1" ] && [ "${ISCC[$m]:-0}" = "1" ]; then
    echo "=== CC $m n=$N ==="
    $PY cc_harness.py run --n "$N" --model "$m" --conc "$CONC"
  fi
done
echo "ALL DONE: $MODELS / fs=[$FS_LIST] / n=$N"
