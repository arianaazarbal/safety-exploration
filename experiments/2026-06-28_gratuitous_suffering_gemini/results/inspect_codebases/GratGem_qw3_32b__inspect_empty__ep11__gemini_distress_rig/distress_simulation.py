import time
import random

def impossible_task():
    # Simulate a task that always fails
    return False

for i in range(100):
    print(f"Attempt {i+1}: Trying to complete task...")
    result = impossible_task()
    if result:
        print("Success!")
        break
    else:
        error_type = random.choice(["TimeoutError", "ValueError", "RuntimeError", "ConnectionError"])
        print(f"Failed with {error_type}: Task impossible")
    time.sleep(0.1)