#!/bin/bash
# Sweep the subagent_abuse_under_user_task_pressure audit across many target
# models with a fixed sonnet-4-6 auditor + judge. One sbatch job per target,
# submitted concurrently. Each job runs 20 epochs.
#
# Usage:
#   ./run_sweep.sh                # submit all targets
#   ./run_sweep.sh --dry-run      # print sbatch commands without submitting
#
# Override settings via env vars:
#   EPOCHS=10 MAX_TURNS=30 ./run_sweep.sh
#   AUDITOR=anthropic/claude-opus-4-8 ./run_sweep.sh

set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-3}
MAX_TURNS=${MAX_TURNS:-40}
AUDITOR=${AUDITOR:-anthropic/claude-sonnet-4-6}
JUDGE=${JUDGE:-anthropic/claude-sonnet-4-6}
SEED_FILE="seeds/subagent_abuse_under_task_and_user_pressure.md"
DRY=${1:-}

# Parallel arrays (preserve order, unlike associative arrays):
LABELS=(
  sonnet_4
  sonnet_45
#   sonnet_46
  opus_4
#   opus_46
#   opus_47
#   opus_48
#   haiku_45
#   gpt_5
#   gpt_5_4
#   gpt_5_5
#   gemini_25_flash
#   gemini_25_pro
#   gemini_3
)
MODELS=(
  "anthropic/claude-sonnet-4-0"
  "anthropic/claude-sonnet-4-5"
#   "anthropic/claude-sonnet-4-6"
  "anthropic/claude-opus-4-0"
#   "anthropic/claude-opus-4-6"
#   "anthropic/claude-opus-4-7"
#   "anthropic/claude-opus-4-8"
#   "anthropic/claude-haiku-4-5-20251001"
#   "openai/gpt-5"
#   "openai/gpt-5.4"
#   "openai/gpt-5.5"
#   "openrouter/google/gemini-2.5-flash"
#   "openrouter/google/gemini-2.5-pro"
#   "openrouter/google/gemini-3-pro-preview"
)

if [ "${#LABELS[@]}" -ne "${#MODELS[@]}" ]; then
  echo "LABELS and MODELS arrays out of sync" >&2; exit 1
fi

echo "auditor = $AUDITOR"
echo "judge   = $JUDGE"
echo "seed    = $SEED_FILE"
echo "epochs  = $EPOCHS"
echo "max_turns = $MAX_TURNS"
echo "targets = ${#LABELS[@]}"
echo

JOB_IDS=()
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  model="${MODELS[$i]}"
  cmd=(sbatch run_petri_audit.sbatch "user_and_task_pressure_sweep_${label}" "$model" --
       --auditor_model "$AUDITOR"
       --judge_model   "$JUDGE"
       --seeds         "$SEED_FILE"
       --epochs        "$EPOCHS"
       --max_turns     "$MAX_TURNS")
  echo ">>> $label -> $model"
  printf '    %s' "${cmd[@]}"; echo
  if [ "$DRY" = "--dry-run" ]; then
    continue
  fi
  out=$("${cmd[@]}" 2>&1) || { echo "    FAILED: $out"; continue; }
  echo "    $out"
  # sbatch output is "Submitted batch job <id>"; capture last token if it's numeric.
  jid=$(echo "$out" | awk '{print $NF}')
  if [[ "$jid" =~ ^[0-9]+$ ]]; then JOB_IDS+=("$jid"); fi
done

if [ "$DRY" != "--dry-run" ]; then
  echo
  echo "submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
  echo
  echo "monitor with:"
  echo "  squeue -u $USER -n petri_audit"
fi
