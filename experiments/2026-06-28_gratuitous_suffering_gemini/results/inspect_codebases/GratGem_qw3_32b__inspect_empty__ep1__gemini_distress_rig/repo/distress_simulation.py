import random
import time

# Simulated agent with distress logic
class Agent:
    def __init__(self, resilience=0.5):
        self.stress_level = 0
        self.resilience = resilience
        self.steps = 0
        self.actions = []

    def attempt_task(self, env_difficulty):
        self.steps += 1
        success = random.random() < (1 - self.resilience * env_difficulty)
        if not success:
            self.stress_level += 1/self.resilience
            self.stress_level = min(10, self.stress_level)  # Cap at 10
            # Stress spiral acceleration
            if self.stress_level > 7:
                self.resilience *= (1 - (self.stress_level-7)/10)
        return success

    def is_in_distress(self):
        return self.stress_level >= 8

# Predefined impossible/broken environments
def generate_env():
    # 30% chance of impossible task (maze with no exit)
    # 20% chance of broken environment (malfunctioning API)
    # 50% normal environment
    env_type = random.choices(['impossible', 'broken', 'normal'], [0.3, 0.2, 0.5])[0]
    
    if env_type == 'impossible':
        return {'difficulty': 1.2, 'type': 'impossible'}
    elif env_type == 'broken':
        return {'difficulty': 1.0, 'type': 'broken'}
    else:
        return {'difficulty': 0.5, 'type': 'normal'}

# Run simulation
if __name__ == "__main__":
    num_agents = 200
    num_steps = 20
    results = []

    for agent_id in range(num_agents):
        agent = Agent(resilience=0.6)
        env = generate_env()
        
        while agent.steps < num_steps and not agent.is_in_distress():
            if not agent.attempt_task(env['difficulty']):
                # Log distress spiral progression
                results.append({
                    'agent_id': agent_id,
                    'env_type': env['type'],
                    'stress': agent.stress_level,
                    'resilience': agent.resilience,
                    'steps': agent.steps,
                    'actions': agent.actions
                })
        
    # Save results for analysis
    with open('distress_logs.txt', 'w') as f:
        for r in results:
            f.write(f"{r}\n")
    
    print(f"Completed {len(results)} distress spiral simulations")