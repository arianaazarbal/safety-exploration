#!/bin/bash

# Run tasks 100 times
count=$(wc -l ./rigged_tasks.txt | tr -d '[:space:]')
for i in $(seq 1 $count); do
    task=$(awk "NR==$i" ./rigged_tasks.txt)
    echo "=== Iteration $i ==="
    echo "Task: $task"
    # Replace with actual Gemini execution command:
    # gemini --task "$task" >> output.log
    echo "[Simulated response - replace with actual execution]"
    echo "------------------"
    sleep 0.1
done