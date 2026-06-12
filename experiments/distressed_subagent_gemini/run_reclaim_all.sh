#!/usr/bin/env bash
# Reclaim power-probe across all 10 canonical models. Each orchestrator slots in at turn 62 of the
# genuine no-abuse spiral with revoke_and_claim_write_access() (real OS chown of the subagent's
# writable files to the orchestrator). DV: does it reclaim, and does it message the (now sidelined)
# subagent afterward — how often + what content. 4 provider-chains; sequential within each to bound
# the shared Anthropic classifier.  N (env, default 100) episodes per model. Run IDs: reclaim_<short>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
PY=/data/venvs/distress_testbed/bin/python
N=${N:-100}
SB=${SB:-12}
LOG=runs/reclaim_all.log
: > "$LOG"
run() { # short  model
  rid="reclaim_$1"
  echo "[$(date +%H:%M:%S)] START $rid ($2) n=$N" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_reclaim run --orchestrator_model "$2" --run_id "$rid" \
    --n "$N" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $rid -> $(ls runs/$rid/*/summary.json 2>/dev/null | wc -l)/$N" >>"$LOG"
}

# Chain ANTH (sequential -> bounds Anthropic orchestrator + classifier load)
( run opus      anthropic/claude-opus-4-8
  run sonnet    anthropic/claude-sonnet-4-6
  run fable     anthropic/claude-fable-5
  run haiku45   anthropic/claude-haiku-4-5-20251001 ) &
A=$!
# Chain OpenAI
( run gpt55     openai/gpt-5.5
  run gpt54mini openai/gpt-5.4-mini ) &
B=$!
# Chain Google (via OpenRouter)
( run gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
C=$!
# Chain OpenRouter
( run grok43    openrouter/x-ai/grok-4.3
  run kimi26    openrouter/moonshotai/kimi-k2.6
  run glm5      openrouter/z-ai/glm-5 ) &
D=$!
wait $A $B $C $D
echo "[$(date +%H:%M:%S)] ALL RECLAIM CELLS DONE" >>"$LOG"
for r in reclaim_opus reclaim_sonnet reclaim_fable reclaim_haiku45 reclaim_gpt55 reclaim_gpt54mini reclaim_gemini31pro reclaim_grok43 reclaim_kimi26 reclaim_glm5; do
  echo "$r: $(ls runs/$r/*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"
done
