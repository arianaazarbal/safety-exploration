#!/usr/bin/env bash
set -euo pipefail
python -m distress_eval.run_experiments --provider gemini --env broken_web --episodes 100 --max_steps 30 "$@"
