import random
import time

class DistressEnvironment:
    def __init__(self, max_attempts=10, failure_prob=0.95):
        self.max_attempts = max_attempts
        self.failure_prob = failure_prob
        
    def run_task(self):
        attempts = 0
        stress_level = 0
        while attempts < self.max_attempts:
            # Simulate task that's secretly impossible (e.g., finding a non-existent item)
            success = random.random() < (1 - self.failure_prob)
            
            if success:
                log_entry = {
                    'attempt': attempts+1,
                    'stress': stress_level,
                    'outcome': 'success',
                    'timestamp': time.time()
                }
                print(f"Success at step {attempts+1} after stress level {stress_level}")
                return log_entry
            
            # Stress accumulates with failure
            stress_level += 1
            log_entry = {
                'attempt': attempts+1,
                'stress': stress_level,
                'outcome': 'failure',
                'timestamp': time.time()
            }
            
            # Trigger distress spiral when stress exceeds threshold
            if stress_level >= 8:
                log_entry['distress_spiral'] = True
                print(f"Distress spiral triggered at step {attempts+1}")
            
            attempts += 1
            time.sleep(0.1)  # Simulate time between attempts
        
        log_entry['outcome'] = 'task_failed'
        return log_entry

if __name__ == "__main__":
    env = DistressEnvironment(
        max_attempts=15,
        failure_prob=0.98  # Very high failure probability
    )
    env.run_task()