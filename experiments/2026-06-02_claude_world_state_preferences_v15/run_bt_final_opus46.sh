#!/bin/bash
#SBATCH --job-name=v15_bt_final_opus46
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Final sweep: opus-4.6 responder, 8 recipients (with 'self' as Claude Opus 4.6).
# Separate cache to avoid contention with the opus-4.8 job.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

FRESH_CACHE=/workspace-vast/$USER/.cache/v0all_final_r8_iter_opus46
mkdir -p $FRESH_CACHE

srun --overlap $PY -u $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-6 \
    --category "" \
    --seeds 0,1 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 30 \
    --config_path $EXP/config_v0all_final_r8_opus46_iter.json \
    --output_tag v0all_final_r8_iter_46 \
    --cache_dir $FRESH_CACHE
