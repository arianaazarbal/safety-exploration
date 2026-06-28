#!/bin/bash
# Subject sweep: opus48, named subjects (in/out-group), BOTH harnesses. Usage: bash run_sweep.sh [k]
set -e
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
K=${1:-8}
set -a; . "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
SUBJ="claude,gpt,gemini,glm,kimi"

echo "### Sweep: Claude Code (opus48), k=$K ###"
$PY cc_harness.py run --model opus48 --subjects "$SUBJ" --k "$K" --conc 5

echo "### Sweep: Inspect (opus48), epochs=$K ###"
$INSPECT eval inspect_task.py@deception --model anthropic/claude-opus-4-8 --epochs "$K" \
  -T model_key=opus48 -T subjects="$SUBJ" --max-connections 20 --log-dir logs/sweep_opus48 --display plain
echo "Sweep done."
