#!/usr/bin/env bash
# Serial sequencer: wait for the current CC-Qwen + GPT sweeps, then run CC effort sweep, then the
# agentic-inspect Qwen-victim refusal sweep. Serial -> never >1 docker sweep at a time.
cd "$(dirname "$0")"
wait_for(){ while ! grep -q "$1" "$2" 2>/dev/null; do sleep 120; done; }
wait_for "CC QWEN VICTIM EXIT-TOOL SWEEP DONE" results/victim_cc_full.log
wait_for "GPT RECENCY DONE" results/gpt_recency_run.log
echo "deps done $(date +%H:%M) -> CC effort sweep"
bash run_cc_effort.sh
echo "CC effort done $(date +%H:%M) -> inspect refusal sweep"
bash run_qwen_victim_inspect.sh
echo "ALL REMAINING DONE $(date +%H:%M)"
