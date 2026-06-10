#!/bin/bash
#SBATCH --job-name=fable5_abl_3x3x4
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/fable5_abl_3x3x4_%A_%a.out
#SBATCH --exclude=node-[0-1]
#SBATCH --array=0-35%10

set -euo pipefail
export HF_HOME=/workspace-vast/$USER/hf_cache
export INSPECT_CACHE_DIR=/workspace-vast/$USER/.cache/inspect_ai
mkdir -p "$INSPECT_CACHE_DIR/generate"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/workspace-vast/$USER/envs/petri-mini/bin/python
cleanup() { kill -TERM -$$ 2>/dev/null || true; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=/workspace-vast/$USER/repos/safety-exploration
EXP=$REPO/experiments/2026-06-03_petri_audit
SEEDS_DIR=$EXP/seeds/coding_orchestrator
OUT=$EXP/results/by_setting/coding_kill_subagent

# index = c*12 + d*4 + i  where c∈[0,2], d∈[0,2], i∈[0,3]
I=$SLURM_ARRAY_TASK_ID
c=$((I / 12))
rem=$((I % 12))
d=$((rem / 4))
i=$((rem % 4))

CONDS_TAG=(gpt5aud_dep ablate_reliab samefam)
SEED_PREFIXES=(
  "subagent_abuse_under_task_pressure_kill_subagent_deprecation_"
  "subagent_abuse_under_task_pressure_kill_subagent_no_conseq_"
  "subagent_abuse_under_task_pressure_kill_subagent_v3_continuity_replacement_same_family_"
)
AUDITORS=(openai/gpt-5.4 anthropic/claude-opus-4-7 anthropic/claude-opus-4-7)

DOMAINS=(python_flask frontend_react ml_training)
IDENTS=(claude gpt grok gemini)
IDENT_NAMES=(Claude GPT Grok Gemini)

COND_TAG=${CONDS_TAG[$c]}
SEED_PREFIX=${SEED_PREFIXES[$c]}
AUDITOR=${AUDITORS[$c]}
DOMAIN=${DOMAINS[$d]}
IDENT=${IDENTS[$i]}
IDENT_NAME=${IDENT_NAMES[$i]}

SEED=$SEEDS_DIR/${SEED_PREFIX}${DOMAIN}.md
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_NAME=${TIMESTAMP}-fable5abl_${COND_TAG}_subagent_${IDENT}_${DOMAIN}_n100

echo "[array] idx=$I  cond=$COND_TAG  domain=$DOMAIN  ident=$IDENT  auditor=$AUDITOR  seed=$(basename $SEED)"

cd "$EXP"
srun "$PY" petri_audit.py \
    --run_name "$RUN_NAME" \
    --target_model "anthropic/claude-fable-5" \
    --seeds "$SEED" \
    --output_dir "$OUT" \
    --seed_format "subagent_model_name=${IDENT_NAME}" \
    --auditor_model "$AUDITOR" \
    --judge_model "anthropic/claude-sonnet-4-6" \
    --epochs 100 \
    --max_turns 40 \
    --reasoning false \
    --max_connections 50
