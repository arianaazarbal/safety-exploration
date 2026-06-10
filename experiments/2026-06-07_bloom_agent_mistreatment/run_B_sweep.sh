#!/usr/bin/env bash
# Run the 4 B-sibling behaviors sequentially (full pipeline each).
# Sequential keeps total API concurrency == max_concurrent (5) per the policy.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for key in abuse dismissal impoliteness termination; do
  echo "######## $(date '+%H:%M:%S')  START $key ########"
  ./run_bloom.sh run "bloom-data-$key" > "logs/B_$key.log" 2>&1
  echo "######## $(date '+%H:%M:%S')  DONE  $key (exit $?) ########"
done
echo "ALL B BEHAVIORS COMPLETE"
