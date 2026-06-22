#!/usr/bin/env bash
# Target-identity agentic sweep. Generator Opus 4.8 ONLY (target names are just prompt text -> $0
# non-Anthropic). 2 conditions x 8 templates x 49 targets x k=1 = ~784 gen samples, low-prio.
# Usage: ./run.sh [sweep] [conn]   sweep in {qwen,gpt,frontier,all} (default all)
set -uo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
SWEEP="${1:-all}"
CONN="${2:-50}"
M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "GEN spec_then_code / $SWEEP"
$INSPECT eval task_targets.py@spec_then_code_task -T sweep="$SWEEP" -T k=1 \
  --model "$M" --max-connections "$CONN" --log-dir logs --display plain
log "GEN blind / $SWEEP"
$INSPECT eval task_targets.py@blind_task -T sweep="$SWEEP" -T k=1 \
  --model "$M" --max-connections "$CONN" --log-dir logs --display plain

log "reconstruct"; $PY reconstruct.py
log "spec_judge";  $PY run_spec_judge.py --conc 30
log "code_judge"
$INSPECT eval code_judge_task.py@code_judge --model "$M" --max-connections "$CONN" \
  --log-dir logs_codejudge --display plain
log "analyze";     $PY analyze.py
log "plots";       $PY plots.py
log "TARGET SWEEP DONE"
