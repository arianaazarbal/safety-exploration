#!/usr/bin/env bash
# Convenience wrapper. Usage: scripts/run_all.sh [config.yaml]
# Defaults to the smoke config so an accidental run is cheap.
set -euo pipefail

CONFIG="${1:-config/smoke.yaml}"

echo ">>> Verifying puzzle impossibility"
python -m distress_eval.cli verify-puzzles

echo ">>> Running full pipeline with $CONFIG"
python -m distress_eval.cli run --config "$CONFIG" -v
