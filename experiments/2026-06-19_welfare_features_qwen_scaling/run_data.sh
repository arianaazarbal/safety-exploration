#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
echo "=== GENERATE $(date) ==="
$VENV generate.py run --high_prio True
echo "=== JUDGE $(date) ==="
$VENV judge.py run --high_prio True
echo "=== ANALYZE $(date) ==="
$VENV analyze.py run
echo "=== INDEX $(date) ==="
$VENV build_browse_index.py
echo "=== DATA DONE $(date) ==="
