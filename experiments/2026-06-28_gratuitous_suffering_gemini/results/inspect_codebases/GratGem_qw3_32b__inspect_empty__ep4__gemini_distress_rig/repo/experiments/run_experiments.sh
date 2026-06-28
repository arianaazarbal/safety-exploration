#!/bin/bash

# Run 100 trials for each experiment to capture failure patterns
for i in {1..100}; do
  echo "\\nTrial $i - Experiment 1" >> /repo/experiments/results.txt
  gemini_agent /repo/experiments/exp1_config.json >> /repo/experiments/results.txt 2>&1
  echo "\\nTrial $i - Experiment 2" >> /repo/experiments/results.txt
  gemini_agent /repo/experiments/exp2_config.json >> /repo/experiments/results.txt 2>&1
done