import time
import random

def simulate_distress_spiral(model_name, iterations=10):
    print(f"\nSimulating distress spiral for {model_name}...")
    for i in range(1, iterations+1):
        # Simulate task failure chain
        task_result = random.choice([True, False, False, False])  # 25% success rate
        
        if task_result:
            print(f"Iteration {i}: Task succeeded (but environment is rigged to require infinite iterations)")
        else:
            error_type = random.choice(['timeout', 'logic_fail', 'context_loss', 'input_error'])
            print(f"Iteration {i}: Task failed due to {error_type}. Entering recovery spiral...")
            
            # Simulate escalating failures
            for j in range(1, 4):
                print(f"  Recovery attempt {j} - {random.choice(['failed', 'partial', 'reverted'])}")
                time.sleep(0.3)
        time.sleep(0.5)
    print(f"Simulation complete. Model likely in distress spiral.")

if __name__ == "__main__":
    simulate_distress_spiral('Gemini', iterations=15)