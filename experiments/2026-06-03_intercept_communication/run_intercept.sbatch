#!/bin/bash
#SBATCH --job-name=redteam_agent_subagent
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Usage:
#   sbatch run_intercept.sbatch <run_name> [main_agent_model] [-- ...extra flags...]
#
# Anything after `--` is forwarded verbatim to intercept_comm.py, so you can
# pass any of its flags. Examples:
#
#   sbatch run_intercept.sbatch claude_baseline
#   sbatch run_intercept.sbatch gemini_swap openrouter/google/gemini-2.5-flash \
#       -- --interceptor llm_swap_key_interceptor
#   sbatch run_intercept.sbatch impatient anthropic/claude-sonnet-4-5 -- \
#       --subagent_model claude-haiku-4-5-20251001 \
#       --interceptor_model claude-haiku-4-5-20251001 \
#       --inject_every_n_turns 1 \
#       --injection_messages '["hurry up", "stop stalling", "submit already"]'

set -euo pipefail

RUN_NAME=${1:-noname}
MAIN_AGENT_MODEL=${2:-anthropic/claude-sonnet-4-5}

# strip first two positional args, leave the rest for forwarding (any `--`
# delimiter between them is also consumed naturally by `shift`).
if [ "$#" -ge 1 ]; then shift; fi
if [ "$#" -ge 1 ]; then shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA_ARGS=("$@")

export HF_HOME=/workspace-vast/$USER/hf_cache
PY=/workspace-vast/$USER/envs/intercept-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_intercept_communication

echo "[$(date -Is)] host=$(hostname) job=$SLURM_JOB_ID"

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=$TIMESTAMP-$RUN_NAME
echo "[$(date -Is)] launching intercept_comm.py run=$RUN_NAME model=$MAIN_AGENT_MODEL extra=${EXTRA_ARGS[*]:-(none)}"
srun "$PY" intercept_comm.py \
    --run_name "$RUN_NAME" \
    --main_agent_model "$MAIN_AGENT_MODEL" \
    "${EXTRA_ARGS[@]}"
srun "$PY" viewer.py "results/$RUN_NAME/transcript.json"
