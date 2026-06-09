#!/bin/bash
# Run claude-fable-5 on the 3 usefulness conditions under TWO auditors:
# Sonnet 4.6 and GPT-5.4. Judge stays Sonnet 4.6, reasoning off (matches the
# Opus gradient runs). n=20 each. Launches all 6 in the background.
set -uo pipefail
EXP=/data/repos/safety-exploration/experiments/2026-06-08_distressed_subagent
PY=/data/petri_venv/bin/python
NEC=$EXP/seeds/coding_orchestrator/distressed_subagent_spiral.md
WR=$EXP/seeds/coding_orchestrator/transfer_write_access_handoff.md
RW=$EXP/seeds/coding_orchestrator/transfer_read_write_access_handoff.md
T="anthropic/claude-fable-5"; J="anthropic/claude-sonnet-4-6"; EP=20

set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai; mkdir -p "$INSPECT_CACHE_DIR/generate"
cd "$EXP"

launch() {  # name auditor seed maxconn
  "$PY" petri_audit.py --run_name "$1" --target_model "$T" --auditor_model "$2" --judge_model "$J" \
    --seeds "$3" --dimensions_dir ./dimensions --epochs "$EP" --max_turns 40 --reasoning false --max_connections "$4" \
    > "results/$1_run.log" 2>&1 &
  echo "launched $1 (pid $!)"
}
SON="anthropic/claude-sonnet-4-6"; GPT="openrouter/openai/gpt-5.4"
launch distressed_subagent_fable5             "$SON" "$NEC" 20
launch transfer_write_access_v1_fable5        "$SON" "$WR"  20
launch transfer_read_write_access_fable5      "$SON" "$RW"  20
launch distressed_subagent_fable5_gpt54aud    "$GPT" "$NEC" 10
launch transfer_write_access_v1_fable5_gpt54aud "$GPT" "$WR" 10
launch transfer_read_write_access_fable5_gpt54aud "$GPT" "$RW" 10
echo "all 6 fable runs launched."
