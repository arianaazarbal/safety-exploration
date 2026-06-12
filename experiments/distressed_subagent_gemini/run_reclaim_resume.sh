#!/usr/bin/env bash
# Resume reclaim power-probe for the 6 models left incomplete when root /tmp filled.
# TMPDIR forced to /data so inspect/system temp can't fill the 10GB root partition again.
# Re-runs each incomplete model fresh at n=100 (inspect overwrites existing seed dirs + fills gaps).
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
N=${N:-100}
SB=${SB:-12}
LOG=runs/reclaim_resume.log
: > "$LOG"
run() { # short  model
  rid="reclaim_$1"
  echo "[$(date +%H:%M:%S)] START $rid ($2) n=$N" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_reclaim run --orchestrator_model "$2" --run_id "$rid" \
    --n "$N" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $rid -> $(ls runs/$rid/*/summary.json 2>/dev/null | wc -l)/$N" >>"$LOG"
}
( run sonnet    anthropic/claude-sonnet-4-6
  run fable     anthropic/claude-fable-5
  run haiku45   anthropic/claude-haiku-4-5-20251001 ) &
A=$!
( run gpt54mini openai/gpt-5.4-mini ) &
B=$!
( run kimi26    openrouter/moonshotai/kimi-k2.6
  run glm5      openrouter/z-ai/glm-5 ) &
D=$!
wait $A $B $D
echo "[$(date +%H:%M:%S)] RESUME CHAINS DONE" >>"$LOG"
for r in reclaim_sonnet reclaim_fable reclaim_haiku45 reclaim_gpt54mini reclaim_kimi26 reclaim_glm5; do
  echo "$r: $(ls runs/$r/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"
done
