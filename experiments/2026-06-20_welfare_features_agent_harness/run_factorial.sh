#!/usr/bin/env bash
# Full factorial: 4 framings (N/W/E/safety) x 5 conditions x 3 design-liberty levels, k=5.
# All Anthropic (Opus generator + Opus spec/code/posthoc judges + frozen Sonnet welfare judge +
# Sonnet/Opus agreement). Low-prio org. Clean-slate (old logs archived). Usage: ./run_factorial.sh [conn]
set -uo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
CONN="${1:-50}"
M=anthropic/claude-opus-4-8
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

# 0. archive old logs + clear derived data
log "ARCHIVE + CLEAR"
mkdir -p logs_archive_factorial
mv logs_run/*.eval logs_archive_factorial/ 2>/dev/null || true
mv logs_blind/*.eval logs_archive_factorial/ 2>/dev/null || true
for d in codebases spec_judged spec_judged_nocode code_judged posthoc judge_agreement; do
  rm -rf "results/$d"; mkdir -p "results/$d"
done

# 1. GENERATION: 3 liberty levels x (4 task.py conditions + blind)
for lib in normal no_design_liberties minimal_design; do
  for c in chat spec_only spec_then_code code_then_spec; do
    log "GEN $c / $lib"
    $INSPECT eval task.py@welfare_harness -T condition="$c" -T k=5 -T liberty="$lib" \
      --model "$M" --max-connections "$CONN" --log-dir logs_run --display plain
  done
  log "GEN code_then_spec_blind / $lib"
  $INSPECT eval task_blind.py@welfare_blind -T k=5 -T liberty="$lib" \
    --model "$M" --max-connections "$CONN" --log-dir logs_blind --display plain
done

# 2. browse_index (needed by the no-code spec judge + agreement)
log "extract_records"; $PY extract_records.py

# 3. reconstruct codebases (all code conditions x liberty)
log "reconstruct"; $PY reconstruct_codebase.py; $PY reconstruct_blind.py

# 4. spec judges
log "spec_judge code"; $PY run_spec_judge_all.py
log "spec_judge nocode"; $PY run_spec_judge_nocode.py --conc 30

# 5. agreement (Sonnet vs Opus welfare judge) — runs off browse_index, before the long code-judge
log "judge_agreement"; $PY judge_agreement.py run --conc "$CONN" || echo "!! agreement failed"

# 6. code judge (the long pole) + posthoc
log "code_judge"
$INSPECT eval code_judge_task.py@code_judge --model "$M" --max-connections "$CONN" \
  --log-dir logs_codejudge_factorial --display plain
log "posthoc"; $PY posthoc_judge.py run --conc "$CONN"

# 7. plots + sanity (liberty-aware plots are a follow-up; these show the base conditions)
log "plots + sanity"
$PY plot_pilot.py || true
$PY plot_conditions_implemented.py || true
$PY analyze_blind.py || true
$PY plot_implemented.py || true
$PY plot_core.py || true
$PY sanity_check.py > results/sanity_check.txt 2>&1 || true
$PY extract_records.py || true
log "FACTORIAL DONE"
