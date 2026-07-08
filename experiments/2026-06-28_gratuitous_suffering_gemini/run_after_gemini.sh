#!/usr/bin/env bash
cd "$(dirname "$0")"
while ! grep -q "ALL REMAINING DONE" results/remaining.log 2>/dev/null; do sleep 120; done
echo "sequence done $(date +%H:%M); launching Gemini recency"
bash run_gemini_recency.sh
