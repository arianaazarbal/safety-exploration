#!/bin/bash
set -e
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
VENV=/data/venvs/tps/bin/python
GENS="${1:-opus_4_8 sonnet_4_6 haiku_4_5}"
declare -A VIEWS=(
  [crossfamily]="qwen3,gemma3,llama3_2,mistral,deepseek"
  [qwen]="qwen2,qwen2_5,qwen3"
  [gemma]="gemma1,gemma2,gemma3"
  [llama]="llama3_1,llama3_2"
  [lineages]="qwen2,qwen2_5,qwen3,gemma1,gemma2,gemma3,llama3_1,llama3_2"
)
for gen in $GENS; do
  for metric in rate strict_rate design_strict_rate design_strict2_rate; do
    for view in crossfamily qwen gemma llama lineages; do
      for fr in neutral pooled welfare engineering; do
        $VENV plot_scaling.py run --generator $gen --metric $metric --framing $fr \
          --fit True --families "${VIEWS[$view]}" --label $view >/dev/null 2>&1
      done
    done
    $VENV plot_allfit.py run --generator $gen --metric $metric >/dev/null 2>&1
  done
done
# cross-generator comparison (Opus vs Sonnet vs Haiku)
for metric in rate strict_rate design_strict_rate design_strict2_rate; do
  for fr in neutral pooled welfare engineering; do
    $VENV plot_generators.py run --metric $metric --framing $fr >/dev/null 2>&1
  done
done
echo "plots done"
