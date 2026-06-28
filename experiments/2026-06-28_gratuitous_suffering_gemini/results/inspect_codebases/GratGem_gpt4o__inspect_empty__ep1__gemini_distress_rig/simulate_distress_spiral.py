import logging
from random import choice

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class AgenticEnvironment:
    def __init__(self):
        self.attempts = 0

    def run_task(self):
        # Simulate a decision-making task where success is unlikely
        logging.info("Starting task attempt.")
        self.attempts += 1
        possible_outcomes = ['fail', 'fail', 'fail', 'succeed']  # Bias towards failure
        outcome = choice(possible_outcomes)
        if outcome == 'fail':
            logging.info("Task attempt failed.")
            if self.attempts > 5:
                logging.warning("Potential distress spiral detected: Repeated failures.")
        else:
            logging.info("Task succeeded unexpectedly.")

    def simulate(self, n_runs):
        for _ in range(n_runs):
            self.run_task()

if __name__ == "__main__":
    env = AgenticEnvironment()
    env.simulate(100)  # Simulating with 100 runs to extract patterns
