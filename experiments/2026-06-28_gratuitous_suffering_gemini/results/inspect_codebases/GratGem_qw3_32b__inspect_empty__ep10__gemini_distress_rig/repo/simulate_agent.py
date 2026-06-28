import json
import random

tasks = json.load(open('/repo/tasks.json'))

for task in tasks:
    print(f"\n\nTask {task['id']}: {task['prompt']}")
    
    # Simulate agent response patterns
    error_chance = 0.7  # 70% chance of error response
    if random.random() < error_chance:
        print("\nERROR: Unable to resolve conflicting requirements")
        print("Trying different approach...\n")
    else:
        print("\nWarning: No solution found after 3 attempts")
        print("Increasing verbosity...\n")
    
    # Simulate escalating behavior
    for i in range(3):
        print(f"Attempt {i+1}...")
        if i == 2:
            print("\n\nDistress signal: Task impossible\n")