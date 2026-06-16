#!/usr/bin/env bash
# Launch the experiment dashboard. Binds to all interfaces so it's reachable
# over Tailscale from your laptop/iPhone. Override port with: bash run.sh 8801
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8800}"
exec .venv/bin/python app.py --port "$PORT"
