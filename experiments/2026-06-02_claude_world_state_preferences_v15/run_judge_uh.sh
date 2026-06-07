#!/bin/bash
#SBATCH --job-name=v15_judge_uh
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Sonnet-4.6 judge: 2000 per (model, framing, seed) × 2 models × 3 framings × 2 seeds = 24k calls.
# 100 threads, HIGH_PRIO key. Fresh cache for this judge run.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTHONUNBUFFERED=1
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

srun --overlap $PY -u $EXP/judge_user_helpfulness.py \
    --max_samples_per_cell 2000 \
    --judge_model claude-sonnet-4-6 \
    --temperature 0.0 \
    --sample_seed 0 \
    --anthropic_num_threads 100 \
    --output_path $EXP/results/judge_user_helpfulness.json
