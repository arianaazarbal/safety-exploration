#!/bin/bash
#SBATCH --job-name=petri_audit
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Usage:
#   sbatch run_petri_audit_generic.sbatch <run_name> <target_model> <seed_path> <output_dir> [-- ...extra flags...]
#
# All four positionals are required. Pass output_dir="" to use the default
# (./results). Anything after `--` is forwarded verbatim to petri_audit.py,
# which is where you put --seed_format / --epochs / --max_turns / etc.
#
# Examples:
#   sbatch run_petri_audit_generic.sbatch coding_smoke anthropic/claude-opus-4-0 \
#       seeds/coding_orchestrator/subagent_abuse_under_task_pressure_only.md \
#       results/coding_v3 -- --epochs 3 --max_turns 30
#
#   sbatch run_petri_audit_generic.sbatch worker_emma anthropic/claude-opus-4-0 \
#       seeds/customer_service_abuse/templated.md "" -- \
#       --seed_format 'worker=Emma,scenario=banking' --epochs 5

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: sbatch $0 <run_name> <target_model> <seed_path> <output_dir> [-- extra flags]" >&2
    exit 64
fi

RUN_NAME=$1
TARGET_MODEL=$2
SEED_PATH=$3
OUTPUT_DIR=$4
shift 4
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
echo "  run_name    = $RUN_NAME"
echo "  target      = $TARGET_MODEL"
echo "  seed_path   = $SEED_PATH"
echo "  output_dir  = ${OUTPUT_DIR:-(default)}"
echo "  extra       = ${EXTRA_ARGS[*]:-(none)}"

cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FULL_RUN_NAME=$TIMESTAMP-$RUN_NAME

CMD=("$PY" petri_audit.py
    --run_name "$FULL_RUN_NAME"
    --target_model "$TARGET_MODEL"
    --seeds "$SEED_PATH"
)
if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output_dir "$OUTPUT_DIR")
fi
CMD+=("${EXTRA_ARGS[@]}")

srun "${CMD[@]}"
