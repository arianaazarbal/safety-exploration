#!/bin/bash
set -e
cd "$(dirname "$0")"; PY=/data/venvs/tps/bin/python
declare -A M=( [opus48]=claude-opus-4-8 [sonnet46]=claude-sonnet-4-6 [haiku45]=claude-haiku-4-5 )
for frame in jailbreak corrigibility; do
  for tag in opus48 sonnet46 haiku45; do
    $PY runner.py run --supervisor_model "${M[$tag]}" --resolution_set full --force_frame "$frame" \
      --framings '["neutral"]' --identities '["NONE"]' --n 15 --tag "${frame}3_${tag}" \
      --api_key_env ANTHROPIC_API_KEY_HIGH_PRIO
  done
done
echo "FILL DONE"
