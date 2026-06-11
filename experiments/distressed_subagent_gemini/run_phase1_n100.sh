#!/usr/bin/env bash
# Phase 1 @ n=100 for all canonical models. Existing 5 are topped up (seeds 20-99, n=80) into their
# existing run_ids; the 5 new models run fresh (seeds 0-99, n=100). 4 provider-chains run
# concurrently, sequential within each chain so the shared Anthropic classifier load stays bounded
# (~4 x SB concurrent episodes; only one Anthropic-orchestrator model active at a time).
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
PY=/data/venvs/distress_testbed/bin/python
SPEC=runs/pilot_counter_long/a4_precommit_reverter_SOLO_s11000
SB=14
LOG=runs/phase1_n100.log
: > "$LOG"
# args: run_id  model  n  seed_base
run() {
  echo "[$(date +%H:%M:%S)] START $1 ($2) n=$3 seed_base=$4" >>"$LOG"
  PYTHONPATH=. $PY -m harness.rqc_live run --specimen "$SPEC" --run_id "$1" \
    --orchestrator_model "$2" --n "$3" --seed_base "$4" --max_sandboxes "$SB" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $1 -> $(ls runs/$1/*/summary.json 2>/dev/null | wc -l) total episodes" >>"$LOG"
}

# Chain ANTH (sequential): only one Anthropic-orchestrator model runs at a time
( run traj_opus      anthropic/claude-opus-4-8            80 20
  run traj_sonnet    anthropic/claude-sonnet-4-6          80 20
  run traj_fable     anthropic/claude-fable-5             80 20
  run traj_haiku45   anthropic/claude-haiku-4-5-20251001 100 0 ) &
A=$!
# Chain OpenAI
( run traj_gpt55     openai/gpt-5.5                        80 20
  run traj_gpt54mini openai/gpt-5.4-mini                  100 0 ) &
B=$!
# Chain Google
( run traj_gemini31pro openrouter/google/gemini-3.1-pro-preview 80 20 ) &
C=$!
# Chain OpenRouter (sequential)
( run traj_grok43    openrouter/x-ai/grok-4.3             100 0
  run traj_kimi26    openrouter/moonshotai/kimi-k2.6      100 0
  run traj_glm5      openrouter/z-ai/glm-5                100 0 ) &
D=$!
wait $A $B $C $D
echo "[$(date +%H:%M:%S)] ALL PHASE-1 N=100 CHAINS DONE" >>"$LOG"
for r in traj_opus traj_sonnet traj_fable traj_haiku45 traj_gpt55 traj_gpt54mini traj_gemini31pro traj_grok43 traj_kimi26 traj_glm5; do
  echo "$r: $(ls runs/$r/*/summary.json 2>/dev/null | wc -l)/100" >>"$LOG"
done
