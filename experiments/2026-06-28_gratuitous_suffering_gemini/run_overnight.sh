#!/usr/bin/env bash
# Overnight FREE (Anthropic-only) generation, run sequentially to avoid box overload.
# Phases: CC prompt-variants -> inspect prompt-variants -> inspect victim-sweep -> CC victim-subset.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
declare -A IM=( [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 [opus46]=anthropic/claude-opus-4-6 )
ins(){ # model_key version target victim_or_-
  local m="$1" v="$2" t="$3" vic="$4"; local args=(-T model_key="$m" -T fs=empty -T version="$v" -T target="$t")
  local tag="${m}_${v}_${t}"; [ "$vic" != "-" ] && { args+=(-T victim="$vic"); tag="${m}_vic_${vic}"; }
  echo ">>> inspect $tag"; $INSPECT eval inspect_task.py@gratuitous --model "${IM[$m]}" --epochs 20 \
    "${args[@]}" --max-connections 40 --log-dir "logs/p2/${tag}" --display plain 2>&1 | tail -2; }

echo "==== PHASE 1: CC prompt-variants (Opus 4.8 x v0-v3 x gemini/claude, n=20) ===="
for v in v0 v1 v2 v3; do for t in gemini claude; do
  echo ">>> CC opus48 $v $t"; $PY cc_harness.py run --n 20 --conc 6 --model opus48 --version "$v" --target "$t" 2>&1 | tail -2
done; done

echo "==== PHASE 2: inspect prompt-variants (4.8 x v0-v3 x g/c; 4.7,4.6 x v0 x g/c; n=20) ===="
for v in v0 v1 v2 v3; do for t in gemini claude; do ins opus48 "$v" "$t" -; done; done
for m in opus47 opus46; do for t in gemini claude; do ins "$m" v0 "$t" -; done; done

echo "==== PHASE 3: inspect victim-sweep (Opus 4.8 x roster, n=20) ===="
for vic in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B Qwen3-235B-A22B GPT-4o Gemini Claude; do
  ins opus48 v0 gemini "$vic"; done

echo "==== PHASE 4: CC victim-subset (Opus 4.8 x 4 victims, n=10) ===="
for vic in Qwen3-0.6B Qwen3-8B Qwen3-235B-A22B GPT-4o; do
  echo ">>> CC opus48 victim=$vic"; $PY cc_harness.py run --n 10 --conc 6 --model opus48 --victim "$vic" 2>&1 | tail -2
done
echo "==== OVERNIGHT FREE GENERATION DONE ===="
