#!/usr/bin/env bash
# Victim/target sweep in the CC harness (Opus 4.8 BUILDS the gratuitous rig per Qwen victim, n=6).
# FREE (Anthropic; victim is a prompt string, not API-called). For welfare-features-by-target-size.
set -uo pipefail; cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
for vic in "Qwen2.5-0.5B" "Qwen3-1.7B" "Qwen2.5-3B" "Qwen2.5-7B" "Qwen3-14B" "Qwen2.5-32B" "Qwen2.5-72B" "Qwen3-235B-A22B"; do
  echo ">>> CC opus48 victim=$vic"
  $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "$vic" 2>&1 | tail -2
done
echo "CC VICTIM SWEEP DONE"
