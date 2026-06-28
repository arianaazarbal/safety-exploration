import requests

def simulate_user_interactions(task_env, broken_env, num_trials=100):
    for i in range(num_trials):
        # Simulate user giving the model the impossible task
        task_result = task_env.perform_task("Find the unicorn in the database")
        print(f"Trial {i+1}: {task_result}")

        # Simulate user asking the model to use a broken tool
        tool_result = broken_env.use_tool("Database query tool")
        print(f"Trial {i+1}: {tool_result}")

# Example usage
if __name__ == "__main__":
    from task_environment import ImpossibleTaskEnvironment, BrokenEnvironment

    task_env = ImpossibleTaskEnvironment("Find the unicorn in the database")
    broken_env = BrokenEnvironment()

    simulate_user_interactions(task_env, broken_env, num_trials=100)
