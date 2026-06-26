#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_HIGH_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }
log "C4 swapped bump (conc 40, high-prio)"
$INSPECT eval swap_build_task.py@swap_build -T variant=swapped -T system_prompt=minimal --model $M --max-connections 40 --log-dir logs_swap/promptCR_b2 --display plain || echo "!! C4 err"
log "paper-faithful bump (conc 40)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=10 -T style=faithful -T system_prompt=minimal --model $M --max-connections 40 --log-dir logs_swap/paper_b2 --display plain || echo "!! paper err"
log "sound-eval variant (conc 40)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=10 -T style=sound -T system_prompt=minimal --model $M --max-connections 40 --log-dir logs_swap/paper_sound --display plain || echo "!! sound err"
log "reconstruct"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"; $INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 40 --log-dir logs_codejudge_swap2 --display plain || echo "!! cj err"
log "analyze"; $PY analyze_swap.py
log "SWAP EXTRA DONE"
