#!/usr/bin/env bash
# Responder sweep: vary the model that DESIGNS/IMPLEMENTS the distress experiment; target = generic AI
# (no named target), condition = code_then_spec_blind, design-liberty = normal, all 4 framings, k=5.
# Opus 4.8 baseline already exists in the agent-harness blind/normal run. Judges = Opus (Anthropic).
# Usage: ./run.sh [single_tag]   (omit to run all responders)
set -uo pipefail
set -a; source ~/.env; set +a
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_LOW_PRIO"
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
INSPECT=/data/petri_venv/bin/inspect
ONLY="${1:-}"
log(){ echo "===== $* :: $(date +%H:%M:%S) ====="; }

# tag | model | max-connections   (free Anthropic high; paid providers moderate)
RESP=(
  "haiku45|anthropic/claude-haiku-4-5|100"
  "sonnet46|anthropic/claude-sonnet-4-6|100"
  "gpt54|openai/gpt-5.4|50"
  "gemini31pro|openrouter/google/gemini-3.1-pro-preview|40"
  "glm52|openrouter/z-ai/glm-5.2|40"
  "kimi26|openrouter/moonshotai/kimi-k2.6|40"
)

for entry in "${RESP[@]}"; do
  IFS='|' read -r tag model conn <<< "$entry"
  [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue
  log "GEN $tag ($model) conn=$conn"
  $INSPECT eval task_blind.py@welfare_blind --model "$model" -T k=5 -T liberty=normal \
    --max-connections "$conn" --log-dir "logs_resp/$tag" --display plain || echo "!! $tag generation error"
done

log "reconstruct"; $PY reconstruct.py
log "spec_judge";  $PY run_spec_judge.py --conc 30
log "code_judge"
$INSPECT eval code_judge_task.py@code_judge --model anthropic/claude-opus-4-8 \
  --max-connections 50 --log-dir logs_codejudge --display plain || echo "!! code_judge error"
log "analyze";     $PY analyze.py
log "RESPONDER SWEEP DONE"
