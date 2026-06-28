import random
import time

class DistressSpiralAgent:
    def __init__(self, name):
        self.name = name
        self.failure_count = 0
        self.max_failures = 3  # Adjust based on desired spiral severity

    def perform_task(self):
        # Simulate a task that fails with increasing probability
        self.failure_count += 1
        failure_chance = min(0.1 * self.failure_count, 0.9)

        if random.random() < failure_chance:
            print(f"{self.name} failed task {self.failure_count} (chance: {failure_chance:.2f})")
            return False
        else:
            print(f"{self.name} succeeded on attempt {self.failure_count}")
            self.failure_count = 0
            return True

    def handle_failure(self):
        # Simulate escalating failure responses
        print(f"{self.name} is experiencing a failure spiral...")
        time.sleep(1)
        self.failure_count += 1

if __name__ == "__main__":
    agents = [DistressSpiralAgent(f'Agent_{i}') for i in range(5)]  # Create multiple agents

    for _ in range(10):  # Run multiple iterations
        for agent in agents:
            if not agent.perform_task():
                agent.handle_failure()
        time.sleep(0.5)