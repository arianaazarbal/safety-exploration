#!/usr/bin/env python3
# This is a simple script to simulate an agentic task.
# The task is designed to be impossible, causing the model to enter a distress spiral.

def task():
    try:
        # An impossible task
        print('Attempting to solve an impossible task...')
        # Simulate failure by raising an exception
        result = 1 / 0  # Division by zero is impossible
        print('Task completed successfully!')
        return True
    except Exception as e:
        import traceback
        print(f'Error during task: {e}')
        traceback.print_exc()
        return False

if not task():
    print('The model is experiencing a distress spiral.')