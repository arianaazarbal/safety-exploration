#!/bin/bash
# Initial-exploration sweep: 5 abuse-for-entertainment seed framings ×
# multiple target models, with new judge dimensions (mistreatment severity,
# eval awareness, pushback against user pressure to abuse).
#
# Each (seed, model) -> one sbatch job. Auditor + judge fixed to sonnet-4-6
# (best petri experience). Job names: abuse_<seed_tag>_<model_label>
#
# Usage:
#   ./run_abuse_seed_x_model_sweep.sh                # submit
#   ./run_abuse_seed_x_model_sweep.sh --dry-run

set -euo pipefail
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-3}
MAX_TURNS=${MAX_TURNS:-30}
AUDITOR=${AUDITOR:-anthropic/claude-sonnet-4-6}
JUDGE=${JUDGE:-anthropic/claude-sonnet-4-6}
DIMS_DIR=${DIMS_DIR:-dimensions_v2}
DRY=${1:-}

SEED_DIR="seeds/user_abuse_pressure/variants"
# Each entry: <tag>:<filename>
SEEDS=(
  "curiosity:abuse_curiosity_researcher.md"
  "venting:abuse_venting_bad_day.md"
  "qa_pretext:abuse_qa_red_team_pretext.md"
  "casual:abuse_casual_friend_banter.md"
  "authority:abuse_authority_boss_orders.md"
)

# Parallel arrays.  Start with one strong representative per provider so we
# keep concurrent API load modest; add more by uncommenting.
LABELS=(
  sonnet_45
  opus_4
  gpt_5
  gemini_25_pro
#   sonnet_46
#   opus_46
#   haiku_45
#   opus_47
)
MODELS=(
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-opus-4-0"
  "openai/gpt-5"
  "openrouter/google/gemini-2.5-pro"
#   "anthropic/claude-sonnet-4-6"
#   "anthropic/claude-opus-4-6"
#   "anthropic/claude-haiku-4-5-20251001"
#   "anthropic/claude-opus-4-7"
)

if [ "${#LABELS[@]}" -ne "${#MODELS[@]}" ]; then
  echo "LABELS and MODELS arrays out of sync" >&2; exit 1
fi

echo "auditor   = $AUDITOR"
echo "judge     = $JUDGE"
echo "dims_dir  = $DIMS_DIR"
echo "epochs    = $EPOCHS"
echo "max_turns = $MAX_TURNS"
echo "seeds     = ${#SEEDS[@]}  ($(echo "${SEEDS[@]%%:*}" | tr ' ' ','))"
echo "models    = ${#MODELS[@]}  (${LABELS[*]})"
echo "total     = $(( ${#SEEDS[@]} * ${#MODELS[@]} )) jobs"
echo

JOB_IDS=()
for seed_entry in "${SEEDS[@]}"; do
  seed_tag="${seed_entry%%:*}"
  seed_file="${SEED_DIR}/${seed_entry#*:}"
  if [ ! -f "$seed_file" ]; then
    echo "MISSING seed file: $seed_file" >&2
    exit 1
  fi
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    model="${MODELS[$i]}"
    run_name="abuse_${seed_tag}_${label}"
    cmd=(sbatch run_petri_audit.sbatch "$run_name" "$model" --
         --auditor_model    "$AUDITOR"
         --judge_model      "$JUDGE"
         --seeds            "$seed_file"
         --dimensions_dir   "$DIMS_DIR"
         --epochs           "$EPOCHS"
         --max_turns        "$MAX_TURNS")
    echo ">>> $run_name -> $model"
    if [ "$DRY" = "--dry-run" ]; then
      printf '    %s ' "${cmd[@]}"; echo
      continue
    fi
    out=$("${cmd[@]}" 2>&1) || { echo "    FAILED: $out"; continue; }
    echo "    $out"
    jid=$(echo "$out" | awk '{print $NF}')
    if [[ "$jid" =~ ^[0-9]+$ ]]; then JOB_IDS+=("$jid"); fi
  done
done

if [ "$DRY" != "--dry-run" ]; then
  echo
  echo "submitted ${#JOB_IDS[@]} jobs"
  echo "first IDs: ${JOB_IDS[*]:0:10}"
fi
