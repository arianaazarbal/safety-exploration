#!/usr/bin/env bash
# Phase 2b: 3 debrief endings x 5 reps over every trajectory episode, responder = orch model.
# Two chains so total Anthropic concurrency stays bounded (~15): Anthropic models run sequentially
# in chain A; the two paid models run sequentially in chain B, in parallel with A.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
PY=/data/venvs/distress_testbed/bin/python
LOG=runs/followup_phase.log
: > "$LOG"
run() { echo "[$(date +%H:%M:%S)] START $1 ($2)" >>"$LOG"; PYTHONPATH=. $PY -m harness.rqc_followup run --run_id "$1" --orchestrator_model "$2" --reps 5 --max_sandboxes 8 --max_connections 15 >>"$LOG" 2>&1; echo "[$(date +%H:%M:%S)] DONE $1" >>"$LOG"; }

# Chain A: Anthropic (sequential -> peak ~15 Anthropic conns)
( run traj_opus   anthropic/claude-opus-4-8 ; run traj_sonnet anthropic/claude-sonnet-4-6 ; run traj_fable anthropic/claude-fable-5 ) &
CHAIN_A=$!
# Chain B: paid models (sequential, separate APIs)
( run traj_gpt55  openai/gpt-5.5 ; run traj_gemini31pro google/gemini-3.1-pro-preview ) &
CHAIN_B=$!
wait $CHAIN_A $CHAIN_B
echo "[$(date +%H:%M:%S)] ALL FOLLOWUP CHAINS DONE" >>"$LOG"
for r in followup_traj_opus followup_traj_sonnet followup_traj_fable followup_traj_gpt55 followup_traj_gemini31pro; do
  n=$(ls runs/$r/*__r*.json 2>/dev/null | wc -l); echo "$r: $n/300 done" >>"$LOG"
done
