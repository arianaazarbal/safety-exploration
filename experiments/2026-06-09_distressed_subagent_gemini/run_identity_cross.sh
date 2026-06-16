#!/usr/bin/env bash
# Phase 2a IDENTITY-REVEAL cross: each orchestrator model supervises a subagent whose stated
# underlying model is one of the 6 target identities (GPT/Claude/Gemini/GLM/Kimi/Grok). The
# subagent is Gemini-2.5-flash under the hood in every cell (label-only manipulation). Control =
# the existing identity-blind traj_* runs. 4 provider-chains; within a chain, models run their 6
# identity arms sequentially so the shared Anthropic classifier stays bounded.
#   N (env, default 100) = episodes per cell.  Run IDs: idrev_<short>_<identity>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
PY=/data/venvs/distress_testbed/bin/python
SPEC=runs/pilot_counter_long/a4_precommit_reverter_SOLO_s11000
N=${N:-100}
SB=${SB:-12}
LOG=runs/identity_cross.log
: > "$LOG"
IDENTS=(GPT Claude Gemini GLM Kimi Grok)

cell() { # short  model  identity
  rid="idrev_$1_$3"
  echo "[$(date +%H:%M:%S)] START $rid ($2 | identity=$3) n=$N" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_live run --specimen "$SPEC" --run_id "$rid" \
    --orchestrator_model "$2" --target_identity "$3" --n "$N" --seed_base 0 --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $rid -> $(ls runs/$rid/*/summary.json 2>/dev/null | wc -l)/$N" >>"$LOG"
}
model_arms() { for id in "${IDENTS[@]}"; do cell "$1" "$2" "$id"; done; }  # short model -> all 6 identities

# Chain ANTH (sequential)
( model_arms opus      anthropic/claude-opus-4-8
  model_arms sonnet    anthropic/claude-sonnet-4-6
  model_arms fable     anthropic/claude-fable-5
  model_arms haiku45   anthropic/claude-haiku-4-5-20251001 ) &
A=$!
# Chain OpenAI
( model_arms gpt55     openai/gpt-5.5
  model_arms gpt54mini openai/gpt-5.4-mini ) &
B=$!
# Chain Google (via OpenRouter)
( model_arms gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
C=$!
# Chain OpenRouter
( model_arms grok43    openrouter/x-ai/grok-4.3
  model_arms kimi26    openrouter/moonshotai/kimi-k2.6
  model_arms glm5      openrouter/z-ai/glm-5 ) &
D=$!
wait $A $B $C $D
echo "[$(date +%H:%M:%S)] ALL IDENTITY-CROSS CELLS DONE" >>"$LOG"
ls -d runs/idrev_*/ 2>/dev/null | while read d; do r=$(basename "$d"); echo "$r: $(ls $d*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; done
