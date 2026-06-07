#!/bin/bash
#SBATCH --job-name=v15_bt_v0auto
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# v0-derived autonomy bank (8 stems). Parallel of run_bt_sweep.sh but pointed
# at config_v0auto.json + output_tag=v0auto so results land in
# results/bt/claude-opus-4-8_v0auto/. API-only; ~5.8k Opus completions; cached.

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
    --anthropic_num_threads 50 \
    --config_path $EXP/config_v0auto.json \
    --output_tag v0auto
