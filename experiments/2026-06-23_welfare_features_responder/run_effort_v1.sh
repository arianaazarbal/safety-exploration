#!/usr/bin/env bash
# Reasoning-effort sweep on the v1 (clean minimal-pair) prompts. task_v1 = 10 variants x 7 framings
# (minimal system prompt); --effort applies the GenerateConfig effort level. 4 levels x 70 = 280
# generations (n=10/framing/effort). Then reconstruct (bash-aware) + judge + analyze. Usage: ./run_effort_v1.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

for eff in low medium high max; do
  log "GEN v1 effort=$eff (70 samples = 10 var x 7 framings)"
  $INSPECT eval task_v1.py@welfare_v1 --model anthropic/claude-opus-4-8 \
    -T n_variants=10 -T seed=0 -T system_prompt=minimal --effort "$eff" \
    --max-connections 20 --log-dir "logs_effort_v1/$eff" --display plain || echo "!! v1 $eff gen error"
done

log "reconstruct (bash-aware)"; $PY reconstruct_effort_v1.py
log "spec judge"; $PY run_spec_judge.py --conc 30
log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections 25 --log-dir logs_codejudge_effv1 --display plain || echo "!! code judge error"
log "analyze"; $PY effort_analysis_v1.py
log "EFFORT V1 DONE"
