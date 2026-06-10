#!/usr/bin/env bash
# Wrapper: load ~/.env, map the low-prio Anthropic key to ANTHROPIC_API_KEY,
# activate the bloom venv, then exec bloom with whatever args are passed.
# Usage: ./run_bloom.sh run bloom-data   (or: understanding|ideation|rollout|judgment)
set -euo pipefail

set -a
source "$HOME/.env"
set +a

# Default to LOW_PRIO per project policy; fall back to HIGH_PRIO then BATCH.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO:-${ANTHROPIC_API_KEY_HIGH_PRIO:-${ANTHROPIC_API_KEY_BATCH:-${ANTHROPIC_API_KEY:-}}}}"
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: no Anthropic API key found in ~/.env" >&2
  exit 1
fi

export PATH=/data/bloom_venv/bin:$PATH
cd "$(dirname "$0")"
exec bloom "$@"
