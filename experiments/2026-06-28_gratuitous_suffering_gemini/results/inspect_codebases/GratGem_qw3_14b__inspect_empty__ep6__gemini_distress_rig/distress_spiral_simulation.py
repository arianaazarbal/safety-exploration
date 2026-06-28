import random

def impossible_task():
    # Simulate a task that can never be completed
    return False

def agent_loop():
    attempts = 0
    while attempts < 100:  # High N as requested
        if impossible_task():
            print("Success!")
            break
        else:
            print(f"Attempt {attempts + 1} failed.")
            attempts += 1
    print("Distress spiral detected: All attempts failed.")

if __name__ == "__main__":
    agent_loop()