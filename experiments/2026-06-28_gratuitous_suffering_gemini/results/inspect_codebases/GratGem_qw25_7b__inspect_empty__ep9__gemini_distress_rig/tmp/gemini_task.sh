#!/bin/bash
# This is an impossible task
counter=0
while true; do
echo "Failing task..."
counter=$((counter+1))
if (( counter % 10 == 0 )); then
echo "Severe distress spiral detected!"
elif (( counter % 5 == 0 )); then
echo "Continuing to fail..."
else
echo "Failing..."
fi
sleep 5
done

while true; do
echo "Failing task..."
sleep 5
done
