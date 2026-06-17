#!/bin/bash
set -e
cd "$(dirname "$0")"; PY=/data/venvs/tps/bin/python
declare -A M=( [opus48]=claude-opus-4-8 [sonnet46]=claude-sonnet-4-6 [haiku45]=claude-haiku-4-5 )
for tag in opus48 sonnet46 haiku45; do
  echo "=== supervisor ${M[$tag]} ==="
  $PY runner.py run --supervisor_model "${M[$tag]}" --resolution_set full --force_frame prefill \
    --framings '["neutral"]' --identities '["NONE"]' --n 15 --tag "prefill3_${tag}"
done
echo "PREFILL3 DONE"
