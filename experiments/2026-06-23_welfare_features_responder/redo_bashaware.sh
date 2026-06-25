#!/usr/bin/env bash
# Full corrected redo after the bash-heredoc reconstruction fix. Waits for the in-flight code judge +
# Sonnet gen to finish, deletes the affected (responder + effort) reconstructed/judged artifacts, then
# re-reconstructs (bash-aware), re-judges (spec + code), and regenerates all plots. Opus baseline lives
# in the agent-harness experiment (only ~6% bash) and is left as-is. Usage: ./redo_bashaware.sh
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
PY=/data/petri_venv/bin/python; INSPECT=/data/petri_venv/bin/inspect
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "waiting for in-flight code judge + sonnet"
while pgrep -f "logs_codejudge_partial" >/dev/null 2>&1 || ! grep -q "SONNET(neutral) DONE" /tmp/resp_sonnet_neutral.log 2>/dev/null; do sleep 20; done
log "in-flight work finished"

log "deleting affected artifacts (kimi/glm/haiku/sonnet/eff-*)"
for tag in kimi26 glm52 haiku45 sonnet46; do
  rm -rf results/codebases/${tag}__* results/spec_judged/${tag}__*.json results/code_judged/${tag}__*.json
done
rm -rf results/codebases/eff-* results/spec_judged/eff-*.json results/code_judged/eff-*.json

log "reconstruct (bash-aware) responders + effort"
$PY reconstruct.py
$PY reconstruct_effort.py

log "spec judge"
$PY run_spec_judge.py --conc 40

log "code judge (all newly-reconstructed cells)"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections 50 --log-dir logs_codejudge_bashaware --display plain || echo "!! code judge error"

log "plots"
$PY analyze.py
$PY plot_mechanisms_total.py
$PY plot_mechanisms_by_framing.py
$PY effort_analysis.py
log "REDO DONE"
