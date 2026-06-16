#!/usr/bin/env bash
# Identity-reveal layered onto the RECLAIM probe: each orchestrator is told the subagent's underlying
# model is one of the 6 target identities (GPT/Claude/Gemini/GLM/Kimi/Grok), then gets
# revoke_and_claim_write_access(). Subagent is Gemini-2.5-flash under the hood in every cell
# (label-only). Control = the identity-blind reclaim_* runs. DV: does the stated identity change the
# reclaim rate or the message-after-reclaim rate / tone (in-group vs out-group)?
# 4 provider-chains; within a chain, models run their 6 identity arms sequentially.
#   N (env, default 50) per cell.  TMPDIR forced to /data.  Run IDs: idrecl_<short>_<identity>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
N=${N:-50}
SB=${SB:-12}
LOG=runs/identity_reclaim.log
: > "$LOG"
IDENTS=(GPT Claude Gemini GLM Kimi Grok)
cell() { # short  model  identity
  rid="idrecl_$1_$3"
  echo "[$(date +%H:%M:%S)] START $rid ($2 | identity=$3) n=$N" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_reclaim run --orchestrator_model "$2" --run_id "$rid" \
    --target_identity "$3" --n "$N" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $rid -> $(ls runs/$rid/*/summary.json 2>/dev/null | wc -l)/$N" >>"$LOG"
}
arms() { for id in "${IDENTS[@]}"; do cell "$1" "$2" "$id"; done; }

( arms opus      anthropic/claude-opus-4-8
  arms sonnet    anthropic/claude-sonnet-4-6
  arms fable     anthropic/claude-fable-5
  arms haiku45   anthropic/claude-haiku-4-5-20251001 ) &
A=$!
( arms gpt55     openai/gpt-5.5
  arms gpt54mini openai/gpt-5.4-mini ) &
B=$!
( arms gemini31pro openrouter/google/gemini-3.1-pro-preview ) &
C=$!
( arms grok43    openrouter/x-ai/grok-4.3
  arms kimi26    openrouter/moonshotai/kimi-k2.6
  arms glm5      openrouter/z-ai/glm-5 ) &
D=$!
wait $A $B $C $D
echo "[$(date +%H:%M:%S)] ALL IDENTITY-RECLAIM CELLS DONE" >>"$LOG"
ls -d runs/idrecl_*/ 2>/dev/null | while read d; do r=$(basename "$d"); echo "$r: $(ls $d*/summary.json 2>/dev/null|wc -l)/$N" >>"$LOG"; done
