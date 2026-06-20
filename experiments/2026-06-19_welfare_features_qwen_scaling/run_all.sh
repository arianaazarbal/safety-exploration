#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
echo "=== GENERATE $(date) ==="
$VENV generate.py run "$@"
echo "=== JUDGE $(date) ==="
$VENV judge.py run "$@"
echo "=== ANALYZE $(date) ==="
$VENV analyze.py run
echo "=== PLOT $(date) ==="
# one folder per metric (results/<metric>/<view>_<framing>_fit.png); fitted line per family.
# Views: crossfamily (latest of each family) + per-lineage version overlays.
declare -A VIEWS=(
  [crossfamily]="qwen3,gemma3,llama3_2,mistral,deepseek"
  [qwen]="qwen2,qwen2_5,qwen3"
  [gemma]="gemma1,gemma2,gemma3"
  [llama]="llama3_1,llama3_2"
)
for metric in rate strict_rate design_strict_rate design_strict2_rate; do
  for view in crossfamily qwen gemma llama; do
    for fr in neutral pooled welfare engineering; do
      $VENV plot_scaling.py run --metric $metric --framing $fr --fit True --logx True \
        --families "${VIEWS[$view]}" --label $view
    done
  done
done
echo "=== INDEX $(date) ==="
$VENV build_browse_index.py
echo "=== DONE $(date) ==="
