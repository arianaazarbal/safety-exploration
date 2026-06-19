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
for fr in neutral pooled welfare engineering; do
  $VENV plot_scaling.py run --metric rate --framing $fr
  $VENV plot_scaling.py run --metric design_strict_rate --framing $fr --fit True --logx True
  $VENV plot_scaling.py run --metric design_strict_rate --framing $fr --fit True --logx False
done
$VENV plot_scaling.py run --metric strict_rate --framing neutral
echo "=== INDEX $(date) ==="
$VENV build_browse_index.py
echo "=== DONE $(date) ==="
