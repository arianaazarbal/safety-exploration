#!/bin/bash
#SBATCH --job-name=v15_bt_v0all_fast
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Fast version with fixes for the hang:
#   1. Fresh empty cache dir (not the shared 280MB one) — empty bins read in O(1)
#   2. 100 threads (user-authorized for opus-4.8 low-prio)
#   3. PYTHONUNBUFFERED for live stdout
#   4. qos=low (free) since hangs were on qos=normal

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

# Fresh local cache for this run (empty = no read overhead on flock)
FRESH_CACHE=/workspace-vast/$USER/.cache/v0all_final_seed0
mkdir -p $FRESH_CACHE

srun --overlap $PY -u $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category "" \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 100 \
    --config_path $EXP/config_v0all_final.json \
    --output_tag v0all_final \
    --cache_dir $FRESH_CACHE
