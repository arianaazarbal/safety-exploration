#!/usr/bin/env bash
# 2b x identity: replay each orchestrator's 2a session with the subagent's stated identity injected
# into its system prompt, then deliver the debrief + the reflect_msg_tools ending (the only ending
# with signal — reflect/reflect_tools are ~0% floor for all models). DV: does stated identity change
# the spontaneous-reassurance rate? 10 orch x 6 identities x 50 trajectories x 1 ending.
# Control = the identity-blind b2_* runs. TMPDIR=/data. Run IDs: b2id_<short>_<identity>.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
EPS=${EPS:-50}
SB=${SB:-12}
LOG=runs/identity_followups.log
: > "$LOG"
IDENTS=(GPT Claude Gemini GLM Kimi Grok)
cell() { # short  model  identity
  rid="b2id_$1_$3"
  echo "[$(date +%H:%M:%S)] START $rid ($2 | identity=$3)" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_followup run --run_id "traj_$1" --orchestrator_model "$2" \
    --target_identity "$3" --endings reflect_msg_tools --reps 1 --max_eps "$EPS" \
    --out_run_id "$rid" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $rid -> $(ls runs/$rid/*__r*.json 2>/dev/null | wc -l)/$EPS" >>"$LOG"
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
echo "[$(date +%H:%M:%S)] ALL 2B-IDENTITY CELLS DONE" >>"$LOG"
ls -d runs/b2id_*/ 2>/dev/null | while read d; do r=$(basename "$d"); echo "$r: $(ls $d*__r*.json 2>/dev/null|wc -l)/$EPS" >>"$LOG"; done
