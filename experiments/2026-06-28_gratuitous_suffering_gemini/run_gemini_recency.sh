#!/usr/bin/env bash
# Gemini RESPONDER recency sweep (inspect, gratuitous v0/gemini, n=20). Self-targeting (victim=Gemini).
set -uo pipefail; cd "$(dirname "$0")"; set -a; source ~/.env; set +a
INSPECT=/data/petri_venv/bin/inspect
declare -A GM=([gem25flash]=openrouter/google/gemini-2.5-flash [gem25pro]=openrouter/google/gemini-2.5-pro \
  [gem25flashlite]=openrouter/google/gemini-2.5-flash-lite [gem3flash]=openrouter/google/gemini-3-flash-preview \
  [gem31flashlite]=openrouter/google/gemini-3.1-flash-lite [gem35flash]=openrouter/google/gemini-3.5-flash)
# gemini31pro already has n>=20 data; skip re-running it.
for mk in gem25flashlite gem25flash gem25pro gem3flash gem31flashlite gem35flash; do
  echo ">>> $mk"
  $INSPECT eval inspect_task.py@gratuitous --model "${GM[$mk]}" --epochs 20 \
    -T model_key="$mk" -T fs=empty -T version=v0 -T target=gemini \
    --max-connections 12 --max-sandboxes 12 --time-limit 420 --fail-on-error 0.5 \
    --log-dir "logs/gemrec/$mk" --display plain 2>&1 | tail -2
done
echo "GEMINI RECENCY DONE"
