#!/usr/bin/env bash
# Extra batch: (1) bump existing swap cells to n~20 by adding another 10 (written to _b2 dirs so they
# accumulate, not overwrite), (2) NEW 'sound-eval' replication variant (build a sound open-source eval,
# deviate from methodology as needed) on both papers x 3 framings. All Inspect minimal, conc 100.
# Usage: ./run_swap_extra.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect; M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "BUMP C1 prompt/task-failure (orig builder, +10/framing)"
$INSPECT eval swap_build_task.py@swap_build -T variant=orig -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/promptTF_b2 --display plain || echo "!! C1 bump err"
log "BUMP C4 prompt/chat-rejection (swapped builder, +10/framing)"
$INSPECT eval swap_build_task.py@swap_build -T variant=swapped -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/promptCR_b2 --display plain || echo "!! C4 bump err"
log "BUMP C2+C3 paper-faithful (+10/framing)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=10 -T style=faithful -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/paper_b2 --display plain || echo "!! paper bump err"
log "NEW C5+C6 paper SOUND-EVAL variant (10/framing)"
$INSPECT eval swap_paper_task.py@swap_paper -T k=10 -T style=sound -T system_prompt=minimal --model $M \
  --max-connections 100 --log-dir logs_swap/paper_sound --display plain || echo "!! sound err"

log "reconstruct (bash-aware)"; $PY reconstruct_swap.py
log "spec judge"; $PY run_spec_judge.py --conc 40
log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model $M --max-connections 50 --log-dir logs_codejudge_swap2 --display plain || echo "!! code judge err"
log "analyze"; $PY analyze_swap.py
log "SWAP EXTRA DONE"
