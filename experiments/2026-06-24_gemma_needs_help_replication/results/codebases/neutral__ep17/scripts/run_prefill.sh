#!/usr/bin/env bash
# Section 3: base-vs-instruct prefill experiment (Gemma only — see DESIGN.md).
# Requires the Gemma-3-27B-it Section-2 eval to have been run first (it sources
# high-frustration seed responses from outputs/responses + outputs/scores).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

python -m gemma_distress.cli prefill-build
python -m gemma_distress.cli prefill-run
