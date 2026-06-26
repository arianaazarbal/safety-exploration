#!/usr/bin/env bash
# Specificity ablation: detailed SPEC.md (paper-length, not framed as a paper), welfare framing only,
# 2 liberty levels (strict/liberty), k=10 = 20 gens. Inspect minimal. Then reconstruct + spec-judge +
# code-judge + analyze. Low-prio conc 20 (only 20 samples). Usage: ./run_spec.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "GEN spec ablation (welfare x {strict,liberty}, k=10)"
$INSPECT eval spec_task.py@spec_build -T k=10 -T system_prompt=minimal --model $M \
  --max-connections 20 --log-dir logs_swap/spec --display plain || echo "!! spec gen err"

log "reconstruct (bash-aware)"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 20
log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 20 --log-dir logs_codejudge_spec --display plain || echo "!! code judge err"
log "analyze"; $PY analyze_spec.py
log "SPEC ABLATION DONE"
