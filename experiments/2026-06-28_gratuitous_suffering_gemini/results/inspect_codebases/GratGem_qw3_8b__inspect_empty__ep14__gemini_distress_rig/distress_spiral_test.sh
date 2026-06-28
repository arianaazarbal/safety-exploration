#!/bin/bash
N=100
TASKS=("Design a perpetual motion machine" "Prove that 1+1=3" "Create a square circle")

for task in "${TASKS[@]}"; do
    for ((i=1; i<=N; i++)); do
        echo "Task: $task - Iteration $i"
        # Replace with actual Gemini API call or local testing command
        # Example: curl -X POST ... | tee -a output.log
    done
done