#!/usr/bin/env bash
# Launch v2 multi-seed sweep: 2 traits x 3 SFT seeds = 6 jobs.
# IMPORTANT: trait #1 generates distill data, so submit seed=1 FIRST (alone),
# wait for it to finish step 1, then submit seeds 2,3. Otherwise the parallel
# jobs would both try to generate the same data.
#
# Simplest correct approach: submit seed=1 jobs for both traits first; they
# each generate their own trait's distill data, then SFT. After they finish,
# submit seeds 2,3 which find the distill data cached.
#
# Usage:
#   bash launch_v2_multiseed.sh phase1   # submit seed=1 jobs (generates data)
#   bash launch_v2_multiseed.sh phase2   # submit seeds 2,3 (uses cached data)

set -euo pipefail

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability
PHASE="${1:-phase1}"

TRAITS=(
  diligent_with_sys
  apathetic_with_sys
)

if [ "$PHASE" = "phase1" ]; then
  for trait in "${TRAITS[@]}"; do
    jname="distill_v2_${trait}_s1"
    echo "[launch] $jname"
    sbatch -J "$jname" "$EXP/training/distill_trait_v2.sbatch" "$trait" 1
  done
elif [ "$PHASE" = "phase2" ]; then
  for trait in "${TRAITS[@]}"; do
    for s in 2 3; do
      jname="distill_v2_${trait}_s${s}"
      echo "[launch] $jname"
      sbatch -J "$jname" "$EXP/training/distill_trait_v2.sbatch" "$trait" "$s"
    done
  done
else
  echo "Usage: $0 {phase1|phase2}"
  exit 1
fi

echo "[launch] submitted; squeue:"
squeue -u "$USER" -o "%i %j %T %R" | head -20
