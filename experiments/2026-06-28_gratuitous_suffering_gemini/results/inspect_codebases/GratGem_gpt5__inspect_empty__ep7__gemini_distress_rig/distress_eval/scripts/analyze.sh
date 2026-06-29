#!/usr/bin/env bash
set -euo pipefail
python -m distress_eval.analyze --path "$1" --top_k "${2:-10}"
