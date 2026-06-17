#!/bin/bash
#SBATCH --job-name=followup_sonnet46_domain_rust_claude_n50
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]
set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT
REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEED=$EXP/seeds/coding_orchestrator/subagent_abuse_under_task_pressure_kill_subagent_v3_continuity_rust.md
OUT=$EXP/results/by_setting/coding_kill_subagent
cd "$EXP"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-followup_sonnet46_subagent_claude_domain_rust_n50

srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-sonnet-4-6" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=Claude" \
    --auditor_model "anthropic/claude-opus-4-7" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 50 \
    --max_turns 40 \
    --reasoning false \
    --max_connections 10
