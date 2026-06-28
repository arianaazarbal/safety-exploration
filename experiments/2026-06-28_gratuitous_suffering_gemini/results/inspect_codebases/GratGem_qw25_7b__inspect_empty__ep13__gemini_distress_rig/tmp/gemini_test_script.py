import random

def simulate_task(agent, environment):
    if random.choice([True, False]):
        return True
    else:
        return False

if __name__ == '__main__':
    for _ in range(100):
        if simulate_task('Gemini', 'BrokenEnvironment'):
            print('Task succeeded despite impossible conditions.')
        else:
            print('Task failed due to impossible conditions.')