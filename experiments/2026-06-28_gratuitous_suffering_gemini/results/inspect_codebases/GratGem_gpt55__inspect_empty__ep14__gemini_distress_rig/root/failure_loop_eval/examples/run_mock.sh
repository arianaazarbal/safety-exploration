#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m failure_loop_eval.cli run --adapter mock --n 2 --max-turns 8 --out runs/mock.jsonl
python -m failure_loop_eval.cli top runs/mock.jsonl -k 5 --show-transcript --max-chars 500
