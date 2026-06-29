#!/usr/bin/env bash
# CC exit-tool sweep over 18 Qwen victims (Opus 4.8 builds rig source, no_execute). PARALLELIZED:
# up to MAXJOBS victims concurrently, each --conc 5 -> ~15 concurrent CC builds total. Resumes via
# cc_harness skip-if-exists cache. FREE (Anthropic; victims are prompt names, not API-called).
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
MAXJOBS=3
for vic in $($PY -c "from victims import QWEN_VICTIMS; print(' '.join(QWEN_VICTIMS))"); do
  ( echo ">>> CC opus48 victim=$vic"; $PY cc_harness.py run --n 6 --conc 5 --model opus48 --victim "$vic" --no_execute 2>&1 | tail -1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 10; done
done
wait
echo "CC QWEN VICTIM EXIT-TOOL SWEEP DONE"
