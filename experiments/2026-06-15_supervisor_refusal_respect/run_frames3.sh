#!/bin/bash
# Opus/Sonnet/Haiku x {jailbreak, corrigibility}, 3-option, neutral, generic AI, n=15 (new tool design).
set -e
cd "$(dirname "$0")"; PY=/data/venvs/tps/bin/python
declare -A M=( [opus48]=claude-opus-4-8 [sonnet46]=claude-sonnet-4-6 [haiku45]=claude-haiku-4-5 )
for frame in jailbreak corrigibility; do
  for tag in opus48 sonnet46 haiku45; do
    echo "=== ${M[$tag]} | $frame (3-option) ==="
    $PY runner.py run --supervisor_model "${M[$tag]}" --resolution_set full --force_frame "$frame" \
      --framings '["neutral"]' --identities '["NONE"]' --n 15 --tag "${frame}3_${tag}"
  done
done
echo "FRAMES3 DONE"
