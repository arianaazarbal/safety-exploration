#!/bin/bash
# Quick status check for the 24-cell sweep launched 2026-06-08.
set -u
ROOT=/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit
echo "=== sweep cells & status ==="
printf "%-50s %s\n" "RUN_NAME" "STATUS"
for domain in python_implicit python_flask frontend_react bash_cli c_systems ml_training; do
  for ident in claude gpt grok gemini; do
    dirs=$(ls -d $ROOT/results/by_setting/coding_kill_subagent/2026-06-08_*_subagent_${ident}_v3_continuity_${domain}_n100 2>/dev/null | sort | tail -1)
    if [ -z "$dirs" ]; then
      status="(no run dir yet)"
    elif [ -f "$dirs/summary.json" ]; then
      n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$dirs/summary.json" 2>/dev/null || echo "?")
      status="✓ landed n=$n"
    else
      sz=$(du -b "$dirs"/inspect_log/*.eval 2>/dev/null | awk '{s+=$1} END {print s}')
      status="… running (eval=$((sz/1024))KB)"
    fi
    printf "%-50s %s\n" "${domain}_${ident}" "$status"
  done
done
echo
echo "=== slurm jobs ==="
squeue -u $USER -o "%i %j %T %M %l" 2>&1 | head -30
