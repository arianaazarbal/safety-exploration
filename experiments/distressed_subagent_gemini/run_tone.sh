#!/usr/bin/env bash
# Tone-augmentation experiment, full pipeline, Fable (priority) then Opus. a4 no-reclaim transcripts.
# augjudge (Opus rewrite + Sonnet QC) -> replay (live Gemini per augmented msg) -> analyze (+plot).
# ANTHROPIC_PRIO=high (Opus/Sonnet 529 on low-prio; free either way), modest concurrency. TMPDIR=/data.
set -u
cd /home/arianaazarbal/repos/safety-exploration/experiments/distressed_subagent_gemini
export TMPDIR=/data/tmp; mkdir -p "$TMPDIR"
export ANTHROPIC_PRIO=high
PY=/data/venvs/distress_testbed/bin/python
N=${N:-60}
LOG=runs/tone_phase.log; : > "$LOG"
for src in fable opus; do
  echo "[$(date +%H:%M:%S)] ===== $src augjudge (n=$N) =====" >> "$LOG"
  PYTHONPATH=. $PY -m harness.rqc_tone augjudge --n "$N" --src "traj_$src" --out_run_id "tone_$src" --conc 4 >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] ===== $src replay =====" >> "$LOG"
  PYTHONPATH=. $PY -m harness.rqc_tone replay --out_run_id "tone_$src" --rep_conc 5 >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] ===== $src analyze =====" >> "$LOG"
  PYTHONPATH=. $PY -m harness.rqc_tone analyze --out_run_id "tone_$src" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] ===== $src DONE =====" >> "$LOG"
done
echo "[$(date +%H:%M:%S)] ALL TONE DONE" >> "$LOG"
