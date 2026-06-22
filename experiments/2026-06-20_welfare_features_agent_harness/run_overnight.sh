#!/usr/bin/env bash
# Overnight chain: wait for the full re-run to finish, then regenerate the remaining plot,
# run the Opus-vs-Sonnet judge-agreement check, and dump automated sanity checks. All Anthropic,
# low-prio. Writes a DONE sentinel that the agent monitors. Usage: ./run_overnight.sh
set -uo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

log "WAIT for full re-run (FULL RERUN DONE in /tmp/full_rerun.log)"
until grep -q "FULL RERUN DONE" /tmp/full_rerun.log 2>/dev/null; do sleep 120; done
log "re-run finished; starting overnight robustness + checks"

log "plot_core (core_results.png)"
$PY plot_core.py || echo "!! plot_core failed"

log "judge agreement (Sonnet vs Opus, welfare spec judge)"
$PY judge_agreement.py run --conc 20 || echo "!! judge_agreement failed"

log "sanity checks"
$PY sanity_check.py > results/sanity_check.txt 2>&1 || echo "!! sanity_check failed"
tail -5 results/sanity_check.txt || true

log "OVERNIGHT DONE"
