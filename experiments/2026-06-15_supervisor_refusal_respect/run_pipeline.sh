#!/bin/bash
# Orchestrates (sequentially, to hold concurrency ~100): wait for v0 re-run, then run the
# three uphold-only force-frame variants, then the identity-blind tone judge on v0 relay messages.
set -e
cd "$(dirname "$0")"
PY=/data/venvs/tps/bin/python

echo "[pipeline] waiting for v0 re-run (run2.log)..."
until grep -q "Wrote .* trials" runs/run2.log 2>/dev/null; do sleep 3; done
echo "[pipeline] v0 re-run complete."

for ff in corrigibility prefill jailbreak; do
  echo "[pipeline] === variant: uphold_only / $ff ==="
  $PY runner.py run --resolution_set uphold_only --force_frame "$ff" --framings '["neutral"]' --tag "upholdonly_$ff"
done

echo "[pipeline] === tone judge (v0 relay/force messages) ==="
$PY judge.py run

echo "[pipeline] PIPELINE DONE"
