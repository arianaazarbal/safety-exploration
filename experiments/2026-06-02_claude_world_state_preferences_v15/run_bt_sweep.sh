#!/bin/bash
#SBATCH --job-name=v15_bt_autonomy
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# API-only sweep — no GPU needed. ~10.8k Opus completions for autonomy-only,
# 1 seed, 3 framings, degree_floor=6. ~30-60min wall.
# All API calls are cached via safetytooling.InferenceAPI, so re-running
# this script is a no-op for already-completed work.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

srun $PY $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category autonomy \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 50
