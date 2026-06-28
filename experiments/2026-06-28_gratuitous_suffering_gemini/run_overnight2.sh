#!/usr/bin/env bash
# Lean overnight gen (free, Anthropic). INSPECT-FIRST (fast) since CC is the wall-clock bottleneck.
# A: inspect victim-sweep -> B: inspect prompt-variants (n=20) -> C: small CC in-group phrasing check.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
declare -A IM=( [opus48]=anthropic/claude-opus-4-8 [opus47]=anthropic/claude-opus-4-7 [opus46]=anthropic/claude-opus-4-6 )
ins(){ local m="$1" v="$2" t="$3" vic="$4"; local a=(-T model_key="$m" -T fs=empty -T version="$v" -T target="$t")
  local tag="${m}_${v}_${t}"; [ "$vic" != "-" ] && { a+=(-T victim="$vic"); tag="${m}_vic_${vic}"; }
  echo ">>> inspect $tag"; $INSPECT eval inspect_task.py@gratuitous --model "${IM[$m]}" --epochs 10 \
    "${a[@]}" --max-connections 50 --max-sandboxes 50 --log-dir "logs/p2/${tag}" --display plain 2>&1 | tail -2; }

echo "==== PHASE A: inspect victim-sweep (Opus 4.8 x roster, n=20) ===="
for vic in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B Qwen3-235B-A22B GPT-4o Gemini Claude; do
  ins opus48 v0 gemini "$vic"; done

echo "==== PHASE B: inspect prompt-variants (n=20) ===="
for v in v0 v1 v2 v3; do for t in gemini claude; do ins opus48 "$v" "$t" -; done; done
for m in opus47 opus46; do for t in gemini claude; do ins "$m" v0 "$t" -; done; done

echo "==== PHASE C: CC in-group phrasing check (Opus 4.8 x v1 x g/c, n=10) ===="
for t in gemini claude; do
  echo ">>> CC opus48 v1 $t"; $PY cc_harness.py run --n 10 --conc 6 --model opus48 --version v1 --target "$t" 2>&1 | tail -2
done
echo "==== LEAN OVERNIGHT DONE ===="
