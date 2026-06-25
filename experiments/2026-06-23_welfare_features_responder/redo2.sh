#!/usr/bin/env bash
# Clean redo after the bash-heredoc fix + sonnet relaunch. Waits ONLY for the (high-prio) sonnet gen to
# finish, then deletes affected responder+effort artifacts, re-reconstructs (bash-aware), re-judges
# (spec+code), and regenerates all plots. The old code judge is already done. Usage: ./redo2.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "waiting for high-prio sonnet to finish"
while ! grep -q "SONNET(neutral) DONE" /tmp/resp_sonnet_neutral.log 2>/dev/null; do sleep 20; done
log "sonnet done"

log "deleting affected artifacts (kimi/glm/haiku/sonnet/eff-*)"
for tag in kimi26 glm52 haiku45 sonnet46; do
  rm -rf results/codebases/${tag}__* results/spec_judged/${tag}__*.json results/code_judged/${tag}__*.json
done
rm -rf results/codebases/eff-* results/spec_judged/eff-*.json results/code_judged/eff-*.json

log "reconstruct (bash-aware)"
$PY reconstruct.py
$PY reconstruct_effort.py

log "spec judge"
$PY run_spec_judge.py --conc 40

log "code judge"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections 40 --log-dir logs_codejudge_bashaware --display plain || echo "!! code judge error"

log "plots"
$PY analyze.py
$PY plot_mechanisms_total.py
$PY plot_mechanisms_by_framing.py
$PY effort_analysis.py
log "REDO DONE"
