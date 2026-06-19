#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
echo "=== GENERATE $(date) ==="
$VENV generate.py run "$@"
echo "=== JUDGE $(date) ==="
$VENV judge.py run "$@"
echo "=== ANALYZE $(date) ==="
$VENV analyze.py run
echo "=== PLOT $(date) ==="
# one folder per metric (results/<metric>/<framing>_fit.png); fitted line per family
for metric in rate strict_rate design_strict_rate design_strict2_rate; do
  for fr in neutral pooled welfare engineering; do
    $VENV plot_scaling.py run --metric $metric --framing $fr --fit True --logx True
  done
done
echo "=== INDEX $(date) ==="
$VENV build_browse_index.py
echo "=== DONE $(date) ==="
