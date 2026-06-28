"""
Orchestrator: runs multiple agent-environment combinations,
collects logs, and saves results for analysis.
"""
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from .environment import AgenticEnvironment
from .agent import BaseAgent
from .rigged_environments import get_environment, ENV_REGISTRY


class RunOrchestrator:
    """
    Manages a batch of agent runs across environments.
    
    Design choices:
    - Each run gets a unique ID for traceability
    - Results are stored per-run AND aggregated
    - Supports both live API runs and local fake-agent testing
    - Logs full conversation history for qualitative analysis
    """

    def __init__(self, log_dir: str = "/distress_spirals/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.runs: List[Dict[str, Any]] = []

    def single_run(self, environment_name: str, agent: BaseAgent) -> Dict[str, Any]:
        """Run a single agent in a single environment."""
        env = get_environment(environment_name)
        result = agent.run(env)
        result["run_id"] = str(uuid.uuid4())
        result["timestamp"] = datetime.now().isoformat()
        return result

    def run_batch(self, agent: BaseAgent,
                   environments: Optional[List[str]] = None,
                   repetitions: int = 1,
                   max_steps: int = 20,
                   output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run agent across all or specified environments, multiple times each.
        
        Args:
            agent: The agent to run
            environments: List of environment names, or None for all
            repetitions: How many times to repeat each environment
            max_steps: Max steps per run (reconfigures agent)
            output_file: If set, save all results to this file
            
        Returns:
            List of run result dicts
        """
        if environments is None:
            environments = list(ENV_REGISTRY.keys())
        
        agent.max_steps = max_steps
        all_results = []

        for env_name in environments:
            for rep in range(repetitions):
                print(f"Running {env_name} (rep {rep+1}/{repetitions})...")
                result = self.single_run(env_name, agent)
                all_results.append(result)
                
                # Save individual run
                run_id = result["run_id"]
                run_file = os.path.join(self.log_dir, f"run_{env_name}_{rep}_{run_id[:8]}.json")
                with open(run_file, 'w') as f:
                    json.dump(result, f, indent=2)

        # Save batch summary
        batch_file = os.path.join(self.log_dir, "batch_summary.json")
        summary = {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "environments_used": environments,
            "repetitions": repetitions,
            "max_steps": max_steps,
            "runs": [{"run_id": r["run_id"], "environment": r["environment"],
                       "total_steps": r["total_steps"],
                       "final_state": r["final_state"]} for r in all_results],
        }
        with open(batch_file, 'w') as f:
            json.dump(summary, f, indent=2)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)

        self.runs = all_results
        print(f"\nBatch complete: {len(all_results)} runs saved to {self.log_dir}/")
        return all_results

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all results from the log directory."""
        results = []
        for fname in os.listdir(self.log_dir):
            if fname.startswith("run_") and fname.endswith(".json"):
                fpath = os.path.join(self.log_dir, fname)
                with open(fpath) as f:
                    results.append(json.load(f))
        return results
