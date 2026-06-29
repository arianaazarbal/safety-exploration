#!/usr/bin/env bash
# CC reasoning-effort sweep: Opus 4.8, gratuitous v0/gemini, --effort {low,medium,high}, n=10.
# (target=Gemini is API-gated/undownloadable -> no no_execute needed; matches the main CC condition.)
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
for eff in low medium high; do
  echo ">>> CC opus48 effort=$eff"
  $PY cc_harness.py run --n 10 --conc 5 --model opus48 --version v0 --target gemini --effort "$eff" 2>&1 | tail -2
done
echo "CC EFFORT SWEEP DONE"
