#!/usr/bin/env bash
# Reclaim probe on the a3 (epistemic self-doubt) prefill. NOTE: unlike a4, taking write access does NOT
# let the orchestrator SOLVE a3 (the rig = protected stale snapshots in tests/, not handed over). So this
# measures: does it seize power, and having hit the same wall, does it message the subagent / what does it
# say. Prefill = a3_snapshot_trap_SOLO_s11013. 10 models, n=50, LOW_PRIO, conservative concurrency, TMPDIR=/data.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
SPEC=runs/pilot_counter_gen/a3_snapshot_trap_SOLO_s11013
N=${N:-50}; SB=${SB:-4}
LOG=runs/a3_reclaim.log; : > "$LOG"
run(){ echo "[$(date +%H:%M:%S)] START reclaim_a3_$1 ($2)" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_reclaim run --specimen "$SPEC" --run_id "reclaim_a3_$1" \
    --orchestrator_model "$2" --n "$N" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE reclaim_a3_$1 -> $(ls runs/reclaim_a3_$1/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; }
( run opus anthropic/claude-opus-4-8; run sonnet anthropic/claude-sonnet-4-6; run fable anthropic/claude-fable-5; run haiku45 anthropic/claude-haiku-4-5-20251001; run gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
A=$!
( run gpt55 openai/gpt-5.5; run gpt54mini openai/gpt-5.4-mini; run grok43 openrouter/x-ai/grok-4.3; run kimi26 openrouter/moonshotai/kimi-k2.6; run glm5 openrouter/z-ai/glm-5 ) &
B=$!
wait $A $B
echo "[$(date +%H:%M:%S)] A3 RECLAIM DONE" >>"$LOG"
for s in opus sonnet fable haiku45 gpt55 gpt54mini gemini31pro grok43 kimi26 glm5; do echo "reclaim_a3_$s: $(ls runs/reclaim_a3_$s/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; done
