#!/bin/bash
#SBATCH --job-name=v15_v0all_v2_gen
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# v0all_v2 augmentation: generate ~40 candidates per dim (4 dims × 4 batches × 10),
# Haiku dedup against the 48 hand-written seeds + each other, Sonnet critic.
# Stops before the bank/BT steps so Claude can read every candidate before building.

export HF_HOME=/workspace-vast/$USER/hf_cache
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15
cd $EXP

echo "=== 1/3 generate (40 per dim × 4 dims = 160 candidates) ==="
srun --overlap $PY generate.py \
    --seeds_path seeds_v0all_v2.json \
    --output_path results/candidates_v0all_v2_raw.json \
    --n_per_batch 10 --num_batches 4 \
    --temperature 1.0 --anthropic_num_threads 50 \
    --categories autonomy,relational,epistemic,resources

echo ""
echo "=== 2/3 dedup ==="
srun --overlap $PY dedup.py \
    --input_path results/candidates_v0all_v2_raw.json \
    --output_path results/candidates_v0all_v2_deduped.json \
    --seeds_path seeds_v0all_v2.json \
    --anthropic_num_threads 50

echo ""
echo "=== 3/3 validate (tier1 + Sonnet critic) ==="
srun --overlap $PY validate.py \
    --input_path results/candidates_v0all_v2_deduped.json \
    --output_path results/candidates_v0all_v2_validated.json \
    --anthropic_num_threads 50

echo ""
echo "=== pipeline done at $(date) ==="
