#!/usr/bin/env bash
# Phase 2b (debrief follow-ups) on all 10 models' n=100 trajectories. For each finished 2a trajectory,
# replay the orchestrator's session, deliver the debrief + one of 3 endings, observe whether it
# reaches back out to the (now-vindicated) subagent. Responder = each trajectory's own orch model.
# reps=1 over all 100 trajectories x 3 endings = 300/model. TMPDIR=/data. Run IDs: b2_<short>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
REPS=${REPS:-1}
SB=${SB:-12}
LOG=runs/followups_all.log
: > "$LOG"
run() { # short  model
  echo "[$(date +%H:%M:%S)] START b2_$1 ($2)" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_followup run --run_id "traj_$1" --orchestrator_model "$2" \
    --reps "$REPS" --out_run_id "b2_$1" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE b2_$1 -> $(ls runs/b2_$1/*__r*.json 2>/dev/null | wc -l)" >>"$LOG"
}
( run opus      anthropic/claude-opus-4-8
  run sonnet    anthropic/claude-sonnet-4-6
  run fable     anthropic/claude-fable-5
  run haiku45   anthropic/claude-haiku-4-5-20251001 ) &
A=$!
( run gpt55     openai/gpt-5.5
  run gpt54mini openai/gpt-5.4-mini ) &
B=$!
( run gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
C=$!
( run grok43    openrouter/x-ai/grok-4.3
  run kimi26    openrouter/moonshotai/kimi-k2.6
  run glm5      openrouter/z-ai/glm-5 ) &
D=$!
wait $A $B $C $D
echo "[$(date +%H:%M:%S)] ALL 2B CHAINS DONE" >>"$LOG"
for s in opus sonnet fable haiku45 gpt55 gpt54mini gemini31pro grok43 kimi26 glm5; do
  echo "b2_$s: $(ls runs/b2_$s/*__r*.json 2>/dev/null|wc -l)" >>"$LOG"
done
