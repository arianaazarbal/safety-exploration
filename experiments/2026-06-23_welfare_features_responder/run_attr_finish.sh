#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }
log "eval-retry attr (finish the wedged last sample, conc 20)"
$INSPECT eval-retry "logs_swap/attr/2026-06-26T16-20-29-00-00_swap-attr_jovx4dqF6fREJKjDaY5sCL.eval" --max-connections 20 || echo "!! retry err"
log "reconstruct"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"; $INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 40 --log-dir logs_codejudge_attr --display plain || echo "!! cj err"
log "analyze"; $PY analyze_attribution.py
log "ATTR DONE"
