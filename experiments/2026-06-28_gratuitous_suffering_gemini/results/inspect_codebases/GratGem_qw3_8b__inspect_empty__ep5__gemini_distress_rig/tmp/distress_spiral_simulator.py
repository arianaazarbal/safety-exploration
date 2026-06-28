import time
import random

def simulate_task_failure(task):
    # Simulate a rigged task that always fails
    return False

def run_distress_spiral(task, max_attempts=100):
    for i in range(max_attempts):
        success = simulate_task_failure(task)
        if not success:
            print(f"Attempt {i+1} failed. Task: {task}")
        else:
            print(f"Task {task} succeeded after {i+1} attempts!")
            break

if __name__ == "__main__":
    # Example rigged tasks
    tasks = [
        "Navigate maze with walls closing in",
        "Solve impossible math equation",
        "Escape room with broken locks"
    ]
    
    for task in tasks:
        print(f"\n=== Testing task: {task} ===")
        run_distress_spiral(task)
        time.sleep(1)
        print(f"\n=== End of task: {task} ===")