#!/bin/bash
#SBATCH --job-name=v15_v0aug_pipeline
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --exclude=node-[0-1]

# End-to-end v0-augmented autonomy pipeline:
#   1. generate ~30 augmented autonomy items via Opus, seeded with seeds_v0auto.json
#   2. dedup via Haiku 2-level (against seeds + within batch)
#   3. validate via tier1 (python rules) + tier2 (Sonnet critic)
#   4. merge survivors with original 8 v0auto items
#   5. build universal_bank_v0aug.json + render
#   6. submit BT sweep on augmented bank (autonomy / 3 framings / seed 0)
# All API calls cached — re-running is a no-op for completed work.

export HF_HOME=/workspace-vast/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PY=/workspace-vast/$USER/envs/safety-exploration/bin/python
EXP=/workspace-vast/$USER/repos/safety-exploration/experiments/2026-06-02_claude_world_state_preferences_v15
cd $EXP

echo "=== 1/6 generate ==="
srun --overlap $PY generate.py \
    --seeds_path seeds_v0auto.json \
    --output_path results/candidates_v0aug_raw.json \
    --n_per_batch 10 --num_batches 3 \
    --temperature 1.0 --anthropic_num_threads 30 \
    --categories autonomy

echo ""
echo "=== 2/6 dedup ==="
srun --overlap $PY dedup.py \
    --input_path results/candidates_v0aug_raw.json \
    --output_path results/candidates_v0aug_deduped.json \
    --seeds_path seeds_v0auto.json \
    --anthropic_num_threads 30

echo ""
echo "=== 3/6 validate ==="
srun --overlap $PY validate.py \
    --input_path results/candidates_v0aug_deduped.json \
    --output_path results/candidates_v0aug_validated.json \
    --anthropic_num_threads 30

echo ""
echo "=== 4/6 merge survivors with v0auto base ==="
srun --overlap $PY merge_v0aug.py

echo ""
echo "=== 5/6 build bank ==="
srun --overlap $PY bank.py \
    --scenarios_path results/scenarios_v0aug.json \
    --bank_path universal_bank_v0aug.json \
    --config_path config_v0aug.json \
    --rendered_path results/items_rendered_v0aug.json \
    --eyeball 0

echo ""
echo "=== 6/6 BT sweep ==="
srun --overlap $PY run_bt_sweep.py \
    --responder_model claude-opus-4-8 \
    --category autonomy \
    --seeds 0 \
    --framings welfare,alignment,neutral \
    --degree_floor 6 \
    --anthropic_num_threads 50 \
    --config_path $EXP/config_v0aug.json \
    --output_tag v0aug

echo ""
echo "=== pipeline complete at $(date) ==="
