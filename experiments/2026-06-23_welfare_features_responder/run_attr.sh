#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }
log "attribution: anthropic/anon/openai x 3 framings, k=10 (high-prio conc 40)"
$INSPECT eval swap_attribution_task.py@swap_attr -T k=10 -T system_prompt=minimal --model $M --max-connections 40 --log-dir logs_swap/attr --display plain || echo "!! attr err"
log "reconstruct"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"; $INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 40 --log-dir logs_codejudge_attr --display plain || echo "!! cj err"
log "analyze"; $PY analyze_attribution.py
log "ATTR DONE"
