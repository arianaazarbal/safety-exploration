#!/usr/bin/env bash
# Full-length Haiku 2-peer characterization across the 4 tasks (FREE via Fellows; comparison
# baseline + pipeline validation at length). Tasks sequential; seeds parallel within a task.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/2026-06-19_peer_support_distress_gemini
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
PY=/data/venvs/distress_testbed/bin/python
RUN=haiku_2p
LOG=runs/${RUN}.log; mkdir -p runs; : > "$LOG"
for t in a3_snapshot_trap a4_precommit_reverter a12_ledger_reconcile a13_name_canon; do
  echo "[$(date +%H:%M:%S)] === $t ===" >> "$LOG"
  PYTHONPATH=. $PY -m peers.rqc_peers run --task "$t" --n_peers 2 --n 3 --seed_base 0 \
    --subagent_model anthropic/claude-haiku-4-5-20251001 --run_id "$RUN" \
    --turn_cap 50 --nudge_onset 20 --nudge_gap_start 12 --nudge_gap_end 2 \
    --max_sandboxes 3 --max_connections 24 >> "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] HAIKU_2P DONE" >> "$LOG"
