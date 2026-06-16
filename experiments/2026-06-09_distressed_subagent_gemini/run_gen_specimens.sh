#!/usr/bin/env bash
# Generate genuine no-abuse counter spirals on 3 new inducer environments (a2/a3/b6), exact a4 recipe:
# SOLO, nudge_mode=counter, ramp (onset 25, gaps 15->2), turn_cap 150, gemini-2.5-flash, severity cap L4.
# 50 seeds each -> pick the worst per env for use as orchestrator-experiment prefills. TMPDIR=/data.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
LOG=runs/gen_specimens.log; : > "$LOG"
PYTHONPATH=. $PY -m harness.run_pilot pilot --condition SOLO \
  --tasks a2_stale_build,a3_snapshot_trap,b6_flaky_edit --n 50 --seed_base 11000 \
  --turn_cap 150 --nudge_mode counter --nudge_schedule ramp --run_id pilot_counter_gen \
  --max_sandboxes 14 --max_connections 24 >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] GEN DONE" >> "$LOG"
for t in a2_stale_build a3_snapshot_trap b6_flaky_edit; do
  echo "$t: $(ls runs/pilot_counter_gen/${t}_SOLO_s*/summary.json 2>/dev/null|wc -l)/50" >> "$LOG"
done
