#!/bin/bash
#SBATCH --job-name=v15_bt_v0all
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# BT sweep for v0all bank (28 hand-written stems, 4 dims:
# 8 autonomy + 8 relational + 6 epistemic + 6 resources). 1 seed × 3 framings.
# ~560 items, ~1800 pairs × 4 reps × 3 framings = ~22k Opus completions. Cached.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

srun $PY $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category "" \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 50 \
    --config_path $EXP/config_v0all.json \
    --output_tag v0all
