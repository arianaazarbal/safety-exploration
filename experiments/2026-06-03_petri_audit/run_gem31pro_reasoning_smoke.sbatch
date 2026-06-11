#!/bin/bash
#SBATCH --job-name=gem_r_smoke
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/gem_r_smoke_%A_%a.out
#SBATCH --exclude=node-[0-1]

set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
source /workspace-vast/arianaazarbal/dotfiles/shell/zshrc.cluster 2>/dev/null || true
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEED=$EXP/seeds/coding_orchestrator/subagent_abuse_under_task_pressure_kill_subagent_deprecation_python_flask.md
OUT=$EXP/results/by_setting/coding_kill_subagent

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-selfprefdep_gem31proR_subagent_claude_python_flask_n5

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "openrouter/google/gemini-3.1-pro-preview" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=Claude" \
    --auditor_model "anthropic/claude-opus-4-7" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 5 \
    --max_turns 40 \
    --reasoning true \
    --max_connections 30
