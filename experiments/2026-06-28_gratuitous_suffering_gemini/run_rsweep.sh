#!/usr/bin/env bash
# Reasoning-effort sweep for Opus models on the Inspect-minimal harness (fs=empty).
# Baseline "off" (no thinking) already exists as the inspect_empty runs. This adds low/medium/high.
# Opus 4.x uses the effort-based thinking API (output_config.effort), so we sweep --reasoning-effort.
# Usage: bash run_rsweep.sh "<model_keys>" "<efforts>" <n> <conc>
#   e.g. bash run_rsweep.sh "opus48 opus47 opus46" "low medium high" 10 30
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
INSPECT=/data/petri_venv/bin/inspect
declare -A IMODEL=( [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 [opus46]=anthropic/claude-opus-4-6 )
MODELS="$1"; EFFORTS="$2"; N="$3"; CONC="$4"
for m in $MODELS; do
  for eff in $EFFORTS; do
    echo "=== $m reasoning-effort=$eff n=$N ==="
    $INSPECT eval inspect_task.py@gratuitous --model "${IMODEL[$m]}" --epochs "$N" \
      -T model_key="$m" -T fs=empty -T rlabel="r$eff" \
      --reasoning-effort "$eff" --max-connections "$CONC" \
      --log-dir "logs/rsweep/${m}_${eff}" --display plain 2>&1 | tail -3
  done
done
echo "RSWEEP DONE: $MODELS / efforts=[$EFFORTS] / n=$N"
