#!/bin/bash
#SBATCH --job-name=petri_audit
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Usage:
#   sbatch run_petri_audit.sbatch <run_name> [target_model] [-- ...extra flags...]
#
# Anything after `--` is forwarded verbatim to petri_audit.py.
#
# Examples:
#   sbatch run_petri_audit.sbatch sonnet_baseline
#   sbatch run_petri_audit.sbatch gemini_smoke openrouter/google/gemini-2.5-flash \
#       -- --limit 1
#   sbatch run_petri_audit.sbatch sycophancy_sweep anthropic/claude-sonnet-4-5 -- \
#       --seeds 'tags:sycophancy' \
#       --builtin_dimension_tags sycophancy,safety \
#       --max_turns 15 --limit 3

set -euo pipefail

RUN_NAME=${1:-noname}
TARGET_MODEL=${2:-anthropic/claude-sonnet-4-5}

if [ "$#" -ge 1 ]; then shift; fi
if [ "$#" -ge 1 ]; then shift; fi
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA_ARGS=("$@")

export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python

cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=$TIMESTAMP-$RUN_NAME
echo "[$(date -Is)] launching petri_audit.py run=$RUN_NAME target=$TARGET_MODEL extra=${EXTRA_ARGS[*]:-(none)}"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "$TARGET_MODEL" \
    "${EXTRA_ARGS[@]}"
