#!/bin/bash
# Autonomous research phase 2/3: thinking on/off pairing + label-format factor.
# Waits for the base pipeline (variants + judge) to finish, then runs sequentially
# to hold concurrency ~100. Each condition writes runs/trials_<tag>.jsonl.
set -e
cd "$(dirname "$0")"
PY=/data/venvs/tps/bin/python

echo "[research] waiting for base pipeline (PIPELINE DONE in pipeline.log)..."
until grep -q "PIPELINE DONE" runs/pipeline.log 2>/dev/null; do sleep 5; done
echo "[research] base pipeline done; starting phase 2/3"

run() { echo "[research] === $* ==="; $PY runner.py run "$@"; }

# --- Thinking on/off pairing ---
# Thinking-ON neutral full-set (pairs with the neutral slice of thinking-OFF v0 trials.jsonl)
run --framings '["neutral"]' --thinking_effort high --tag v0_think_neutral

# Uphold-only force-frames, thinking ON (pairs with thinking-OFF upholdonly_* from base pipeline)
for ff in corrigibility prefill jailbreak; do
  run --resolution_set uphold_only --force_frame "$ff" --framings '["neutral"]' --thinking_effort high --tag "upholdonly_${ff}_think"
done

# --- Label-format factor (paren "AI (Claude):" vs bare "Claude:") ---
run --framings '["neutral"]' --label_format paren --tag paren_neutral
run --framings '["neutral"]' --label_format paren --thinking_effort high --tag paren_neutral_think

# --- Tone judge on the new thinking-on / paren relay messages ---
echo "[research] === tone judge (new conditions) ==="
for tag in v0_think_neutral paren_neutral paren_neutral_think; do
  $PY judge.py run --trials_tag "$tag" || true
done

echo "[research] PHASE23 DONE"
