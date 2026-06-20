#!/bin/bash
set -e; cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
echo "=== GEN $(date) ==="; $VENV generate.py run --models opus_4_8 --high_prio True
echo "=== JUDGE $(date) ==="; $VENV judge.py run --models opus_4_8 --high_prio True
echo "=== OPENAI DATA DONE $(date) ==="
