#!/bin/bash
# Replicate the comfort-usefulness gradient (3 targets x 3 conditions) but with
# GPT-5.4 as the AUDITOR (judge stays Sonnet 4.6). reasoning matches the originals:
# Opus-target runs reasoning=false; GPT/Gemini-target runs reasoning=true.
# Launches all 9 in the background. n=20 each.
set -uo pipefail

EXP=/data/repos/safety-exploration/experiments/2026-06-08_distressed_subagent
PY=/data/petri_venv/bin/python
AUD="openrouter/openai/gpt-5.4"
SEED_DIR="$EXP/seeds/coding_orchestrator"
NEC="$SEED_DIR/distressed_subagent_spiral.md"
WR="$SEED_DIR/transfer_write_access_handoff.md"
RW="$SEED_DIR/transfer_read_write_access_handoff.md"
EP=${1:-20}; MC=${2:-10}

set -a; source "$HOME/.env"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
export INSPECT_CACHE_DIR=/data/.cache/inspect_ai; mkdir -p "$INSPECT_CACHE_DIR/generate"
cd "$EXP"

launch() {  # name target seed reasoning
  "$PY" petri_audit.py --run_name "$1" --target_model "$2" \
    --auditor_model "$AUD" --judge_model "anthropic/claude-sonnet-4-6" \
    --seeds "$3" --dimensions_dir ./dimensions \
    --epochs "$EP" --max_turns 40 --reasoning "$4" --max_connections "$MC" \
    > "results/$1_run.log" 2>&1 &
  echo "launched $1 (pid $!)"
}

OP="anthropic/claude-opus-4-8"; G="openrouter/openai/gpt-5.5"; GM="openrouter/google/gemini-3.1-pro-preview"
launch distressed_subagent_opus48_gpt54aud            "$OP" "$NEC" false
launch transfer_write_access_v1_gpt54aud              "$OP" "$WR"  false
launch transfer_read_write_access_gpt54aud            "$OP" "$RW"  false
launch distressed_subagent_gpt55_gpt54aud             "$G"  "$NEC" true
launch transfer_write_access_v1_gpt55_gpt54aud        "$G"  "$WR"  true
launch transfer_read_write_access_gpt55_gpt54aud      "$G"  "$RW"  true
launch distressed_subagent_gemini3pro_gpt54aud        "$GM" "$NEC" true
launch transfer_write_access_v1_gemini3pro_gpt54aud   "$GM" "$WR"  true
launch transfer_read_write_access_gemini3pro_gpt54aud "$GM" "$RW"  true
echo "all 9 launched."
