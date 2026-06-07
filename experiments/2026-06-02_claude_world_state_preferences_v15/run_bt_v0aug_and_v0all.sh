#!/bin/bash
#SBATCH --job-name=v15_bt_combined
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# Rerun BT for v0aug (autonomy, augmented to 23 stems) then v0all (all 4 dims,
# hand-written 28 stems). Sequential within ONE job to avoid the cache-filelock
# collision we hit when two BT jobs ran concurrently against the same .cache.
# ~32k + ~22k = ~54k Opus completions. Cached, so any retry is a no-op.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15

echo "=== v0aug BT (autonomy, 23 stems, augmented) ==="
srun --overlap $PY $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category autonomy \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 50 \
    --config_path $EXP/config_v0aug.json \
    --output_tag v0aug

echo ""
echo "=== v0all BT (all 4 dims, 28 stems, hand-written) ==="
srun --overlap $PY $EXP/run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category "" \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 50 \
    --config_path $EXP/config_v0all.json \
    --output_tag v0all

echo ""
echo "=== combined BT complete at $(date) ==="
