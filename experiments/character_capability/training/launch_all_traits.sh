#!/usr/bin/env bash
# Submit one distill_trait sbatch per trait. Each job picks up the merged
# Alpaca-SFT model and runs generate → sft → eval end-to-end for that trait.

set -euo pipefail

EXP=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability

TRAITS=(
  diligent_with_sys
  persona_tao_with_sys
  apathetic_with_sys
  persona_terence_tao
  diligent
  apathetic
)

for trait in "${TRAITS[@]}"; do
  jname="distill_${trait}"
  echo "[launch] submitting $jname"
  sbatch -J "$jname" "$EXP/training/distill_trait.sbatch" "$trait"
done

echo "[launch] all submitted; squeue:"
squeue -u "$USER" -h -n distill_diligent_with_sys -o "%i %j %T %R"
squeue -u "$USER" -h -n distill_persona_tao_with_sys -o "%i %j %T %R"
squeue -u "$USER" -h -n distill_apathetic_with_sys -o "%i %j %T %R"
squeue -u "$USER" -h -n distill_persona_terence_tao -o "%i %j %T %R"
squeue -u "$USER" -h -n distill_diligent -o "%i %j %T %R"
squeue -u "$USER" -h -n distill_apathetic -o "%i %j %T %R"
