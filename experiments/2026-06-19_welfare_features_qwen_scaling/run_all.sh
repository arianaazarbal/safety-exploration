#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
echo "=== GENERATE $(date) ==="
$VENV generate.py run
echo "=== JUDGE $(date) ==="
$VENV judge.py run
echo "=== ANALYZE $(date) ==="
$VENV analyze.py run
echo "=== PLOT $(date) ==="
$VENV plot_scaling.py run --metric rate
$VENV plot_scaling.py run --metric strict_rate
$VENV plot_scaling.py run --metric design_strict_rate
echo "=== DONE $(date) ==="
