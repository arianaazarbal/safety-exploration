#!/usr/bin/env bash
# Method x Format x Framing swap, all Inspect minimal, conc 100. Matched 10 variants (seed=0).
# C1 prompt/task-failure = EXISTING v1 (neutral/welfare/safety; no re-run). NEW:
#  C4 prompt/chat-rejection (swapped builder) x 3 framings x 10 = 30
#  C2 paper/chat + C3 paper/task (replication) x 3 framings x k=10 = 60
# Then reconstruct + judge + 2x2x3 analysis. Usage: ./run_swap.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "C4 prompt/chat-rejection (swapped builder, 3 framings x 10 = 30)"
$INSPECT eval swap_build_task.py@swap_build -T variant=swapped -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/promptCR --display plain || echo "!! C4 err"
log "C2+C3 paper replication (oggemma+swapped, 3 framings, k=10 = 60)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=10 -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/paper --display plain || echo "!! paper err"

log "reconstruct (bash-aware)"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 50 --log-dir logs_codejudge_swap --display plain || echo "!! code judge err"
log "analyze 2x2x3 (C1 = existing v1)"; $PY analyze_swap.py
log "SWAP DONE"
