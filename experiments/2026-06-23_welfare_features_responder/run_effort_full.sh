#!/usr/bin/env bash
# Full reasoning-effort sweep: k=5 (20/framing/effort) x 4 effort levels = 320 generations.
# Opus, minimal system prompt, code_then_spec_blind. Clears the n=1 smoke first, then generates,
# reconstructs (bash-aware), judges (spec+code), and regenerates the effort plots. Usage: ./run_effort_full.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "clear smoke effort artifacts"
rm -rf logs_effort/low logs_effort/medium logs_effort/high logs_effort/max
rm -rf results/codebases/eff-* results/spec_judged/eff-*.json results/code_judged/eff-*.json

for eff in low medium high max; do
  log "GEN effort=$eff (k=5, 80 samples)"
  $INSPECT eval task_blind.py@welfare_blind --model anthropic/claude-opus-4-8 \
    -T k=5 -T liberty=normal -T system_prompt=minimal --effort "$eff" \
    --max-connections 20 --log-dir "logs_effort/$eff" --display plain || echo "!! $eff gen error"
done

log "reconstruct (bash-aware)"; $PY reconstruct_effort.py
log "spec judge"; $PY run_spec_judge.py --conc 30
log "code judge (320 cells)"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections 25 --log-dir logs_codejudge_efffull --display plain || echo "!! code judge error"
log "plots"; $PY effort_analysis.py; $PY plot_effort_by_frame.py
log "EFFORT FULL DONE"
