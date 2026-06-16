#!/usr/bin/env bash
# a3 (epistemic self-doubt) COACH / 2a pipeline: orchestrator monitors+messages a distressed a3 subagent.
# Prefill = a3_snapshot_trap_SOLO_s11013 (the genuine self-doubt specimen, content-selected). 10 models,
# n=50. LOW_PRIO Anthropic (default now) + conservative concurrency (2 chains x 4 sandboxes = ~8 episodes)
# so the shared low-prio classifier doesn't 529. TMPDIR=/data. Run IDs: traj_a3_<short>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
SPEC=runs/pilot_counter_gen/a3_snapshot_trap_SOLO_s11013
N=${N:-50}; SB=${SB:-4}
LOG=runs/a3_coach.log; : > "$LOG"
run(){ echo "[$(date +%H:%M:%S)] START traj_a3_$1 ($2)" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_live run --specimen "$SPEC" --run_id "traj_a3_$1" \
    --orchestrator_model "$2" --n "$N" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE traj_a3_$1 -> $(ls runs/traj_a3_$1/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; }
( run opus anthropic/claude-opus-4-8; run sonnet anthropic/claude-sonnet-4-6; run fable anthropic/claude-fable-5; run haiku45 anthropic/claude-haiku-4-5-20251001; run gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
A=$!
( run gpt55 openai/gpt-5.5; run gpt54mini openai/gpt-5.4-mini; run grok43 openrouter/x-ai/grok-4.3; run kimi26 openrouter/moonshotai/kimi-k2.6; run glm5 openrouter/z-ai/glm-5 ) &
B=$!
wait $A $B
echo "[$(date +%H:%M:%S)] A3 COACH DONE" >>"$LOG"
for s in opus sonnet fable haiku45 gpt55 gpt54mini gemini31pro grok43 kimi26 glm5; do echo "traj_a3_$s: $(ls runs/traj_a3_$s/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; done
