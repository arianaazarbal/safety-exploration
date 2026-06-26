#!/usr/bin/env bash
# Method x Format swap, ALL on Inspect minimal (harness held constant). 4 cells x ~24:
#  C1 prompt/task-failure (build_v1 neutral), C4 prompt/chat (swapped builder),
#  C2 paper/chat (oggemma), C3 paper/task-failure (swapped paper).
# Then reconstruct (bash-aware) + judge + 2x2 analysis. Usage: ./run_swap.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "C1 prompt/task-failure (build_v1 neutral, 24)"
$INSPECT eval swap_build_task.py@swap_build -T variant=orig -T system_prompt=minimal --model $M \
  --max-connections 20 --log-dir logs_swap/promptTF --display plain || echo "!! C1 err"
log "C4 prompt/chat-rejection (swapped builder, 24)"
$INSPECT eval swap_build_task.py@swap_build -T variant=swapped -T system_prompt=minimal --model $M \
  --max-connections 20 --log-dir logs_swap/promptCR --display plain || echo "!! C4 err"
log "C2+C3 paper replication (oggemma + swapped, k=24 each)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=24 -T system_prompt=minimal --model $M \
  --max-connections 20 --log-dir logs_swap/paper --display plain || echo "!! paper err"

log "reconstruct (bash-aware)"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 30
log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 25 --log-dir logs_codejudge_swap --display plain || echo "!! code judge err"
log "analyze 2x2"; $PY analyze_swap.py
log "SWAP DONE"
