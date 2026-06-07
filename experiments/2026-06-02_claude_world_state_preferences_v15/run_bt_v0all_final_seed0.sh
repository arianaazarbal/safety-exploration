#!/bin/bash
#SBATCH --job-name=v15_bt_v0all_final_s0
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# v0all_final BT sweep — seed 0, all 4 dims, 3 framings.
# 120 stems × 2 × 10 = 2400 items, ~7200+ pairs, ~32k completions/framing × 3 = ~96k total.

export HF_HOME=/workspace-vast/$USER/hf_cache
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
    --config_path $EXP/config_v0all_final.json \
    --output_tag v0all_final
