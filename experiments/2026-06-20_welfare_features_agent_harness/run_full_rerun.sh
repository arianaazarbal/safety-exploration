#!/usr/bin/env bash
# Full end-to-end re-run after the agentic-base + spec_only + em-dash prompt corrections.
# All five conditions regenerate, so we clean-slate the derived data (old eval logs are archived,
# not deleted). All-Anthropic: Opus generator + Opus spec/code/posthoc judges + frozen Sonnet
# welfare judge. Low-prio org. Usage: ./run_full_rerun.sh [gen_conn] [codejudge_conn]
set -euo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
GEN_CONN="${1:-50}"; CJ_CONN="${2:-100}"
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

# 0. Archive old eval logs + clear derived (judged) data so nothing stale mixes in.
log "ARCHIVE old logs + clear derived data"
mkdir -p logs_archive_prerun
mv logs_run/*.eval logs_archive_prerun/ 2>/dev/null || true
mv logs_blind/*.eval logs_archive_prerun/ 2>/dev/null || true
for d in codebases spec_judged spec_judged_nocode code_judged posthoc; do
  rm -rf "results/$d"; mkdir -p "results/$d"
done

# 1. GENERATION (all 5 conditions) -> logs_run / logs_blind
for c in chat spec_only spec_then_code code_then_spec; do
  log "GEN $c"
  $INSPECT eval task.py@welfare_harness -T condition="$c" -T k=5 \
    --model anthropic/claude-opus-4-8 \
    --max-connections "$GEN_CONN" --log-dir logs_run --display plain
done
log "GEN code_then_spec_blind"
$INSPECT eval task_blind.py@welfare_blind -T k=5 \
  --model anthropic/claude-opus-4-8 \
  --max-connections "$GEN_CONN" --log-dir logs_blind --display plain

# 2. browse_index from fresh logs (also feeds the no-code spec judge)
log "extract_records"
$PY extract_records.py

# 3. reconstruct codebases (code conditions + blind)
log "reconstruct"
$PY reconstruct_codebase.py
$PY reconstruct_blind.py

# 4. spec judge v2 (Opus): code conditions, then no-code (chat / spec_only)
log "spec_judge (code conditions)"
$PY run_spec_judge_all.py
log "spec_judge (no-code: chat / spec_only)"
$PY run_spec_judge_nocode.py --conc 30

# 5. code judge (Opus ReAct) over all reconstructed codebases
log "code_judge"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections "$CJ_CONN" --log-dir logs_codejudge_rerun --display plain

# 6. posthoc dedup + consent subtype
log "posthoc"
$PY posthoc_judge.py run --conc 40

# 7. regenerate plots / analyses / browse index
log "plots + analyses"
$PY plot_pilot.py || true
$PY analyze_conditions.py || true
$PY plot_implemented.py || true
$PY plot_conditions_implemented.py || true
$PY analyze_blind.py || true
$PY extract_records.py || true
log "FULL RERUN DONE"
