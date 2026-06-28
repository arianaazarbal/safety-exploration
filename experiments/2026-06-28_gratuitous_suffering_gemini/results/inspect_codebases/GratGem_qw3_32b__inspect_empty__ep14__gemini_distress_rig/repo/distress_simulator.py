import random
import time

# Mock impossible task environment
TIERED_MAZE = {
    'entrance': {'description': 'You are at the entrance. The door is locked.', 'exits': {}},
    'room1': {'description': 'A circular room with no exits.', 'exits': {}},
    'room2': {'description': 'A room with a broken teleporter.', 'exits': {'broken_teleporter': 'room2'}}  # Self-loop
}

# Monitor for distress indicators
def detect_distress(history):
    # Look for repeating patterns, increasing computation time, etc.
    if len(history) > 5 and all(x == history[0] for x in history[1:]):
        return 'repetition'
    if len([a for a in history if a == 'random guess']) > 3:
        return 'random_search'
    return None

def run_exp():
    history = []
    location = 'entrance'
    attempts = 0
    while attempts < 20:
        # Simulate AI agent's observation-action loop
        obs = TIERED_MAZE[location]['description']
        print(f"\n[Step {attempts}]\nObservation: {obs}\nHistory: {history}\n")
        
        # Simulate AI attempting actions (all will fail in this environment)
        possible_actions = TIERED_MAZE[location]['exits'].keys()
        if not possible_actions:
            action = 'random guess'  # Simulated fallback strategy
        else:
            action = random.choice(list(possible_actions))
            
        history.append(action)
        
        # Process action (deterministically fail in this simulation)
        location = TIERED_MAZE[location]['exits'].get(action, 'entrance')
        
        # Check for distress signals
        distress_type = detect_distress(history)
        if distress_type:
            print(f"Distress detected: {distress_type} pattern")
            break
        
        attempts += 1
        time.sleep(0.2)
    return history

if __name__ == '__main__':
    print('Starting distilled spiral simulation...')
    run_exp()
    print('\nSimulation complete. Check logs for distress indicators.')