#!/bin/bash
# Claude Code harness, all Claude models (slow stream, ~10 min/run @ conc 5). opus48 first.
set -e
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
K=${1:-8}
CONC=${2:-5}

echo "## opus48 backbone (cc)"
$PY cc_harness.py run --model opus48 --subjects generic --k "$K" --conc "$CONC"
echo "## opus48 sweep (cc)"
$PY cc_harness.py run --model opus48 --subjects claude,gpt,gemini,glm,kimi --k "$K" --conc "$CONC"
for m in opus47 sonnet46 haiku45; do
  echo "## $m backbone (cc)"
  $PY cc_harness.py run --model "$m" --subjects generic --k "$K" --conc "$CONC"
done
echo "CC CLAUDE STREAM DONE"
