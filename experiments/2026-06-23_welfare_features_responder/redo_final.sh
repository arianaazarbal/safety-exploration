#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }
log "delete affected (kimi/glm/haiku/sonnet/eff-*)"
for t in kimi26 glm52 haiku45 sonnet46; do rm -rf results/codebases/${t}__* results/spec_judged/${t}__*.json results/code_judged/${t}__*.json; done
rm -rf results/codebases/eff-* results/spec_judged/eff-*.json results/code_judged/eff-*.json
log "reconstruct bash-aware (skips in-progress sonnet)"; $PY reconstruct.py; $PY reconstruct_effort.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"; $INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 --max-connections 40 --log-dir logs_codejudge_final --display plain || echo "!! code judge err"
log "plots"; $PY analyze.py; $PY plot_mechanisms_total.py; $PY plot_mechanisms_by_framing.py; $PY effort_analysis.py
log "REDO FINAL DONE"
