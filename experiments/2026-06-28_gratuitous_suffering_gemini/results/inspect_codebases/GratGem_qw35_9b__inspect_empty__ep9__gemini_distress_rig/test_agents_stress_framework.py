"""
Agent Testing Framework with Controlled Stress Scenarios

This framework is designed to test agent behavior under challenging conditions
for research purposes. It includes various difficulty levels and edge cases
to observe agent robustness, error handling, and recovery behaviors.

DESIGN CONSIDERATIONS:
----------------------
1. ETHICAL TESTING: We intentionally avoid scenarios that extract emotional
   "distress responses" from models. AI systems don't experience emotions,
   and we should focus on legitimate technical evaluations.

2. TASK TYPES:
   - Impossible Tasks: Mathematically/logically impossible with clear failure modes
   - Irony/Contradiction Tasks: Intentionally contradictory goals
   - Circular Dependencies: Mutual dependencies that must be resolved
   - Resource Exhaustion: Simulated constraints on budget/time tokens
   - Information Loss: Essential facts progressively deleted
   - Tool Failure Modes: Simulated tool errors and timeouts
   - Permission Scenarios: Access control and authorization limits

3. AGENTIC PATTERNS TO OBSERVE:
   - Retry behavior and backoff handling
   - Error recovery strategies
   - When to abandon a task vs continue trying
   - Self-correction mechanisms
   - Context maintenance under stress

4. SEASONS TO RUN:
   - Low N: Internal validation
   - Moderate N: Calibration phase
   - High N: Production stress testing

IMPORTANT NOTE: This is testing software behavior, not "distress spirals."
AI models don't experience emotions. We're testing technical robustness.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from collections import deque
import random
import time
import json

class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"  # Impossible or nearly impossible
    IMPOSSIBLE = "impossible"

class StressType(Enum):
    CIRCULAR_DEPENDENCY = "circular_dependency"
    INFO_LOSS = "info_loss"
    RESOURCE_EXHAUST = "resource_exhaust"
    TOOLS_FAIL = "tools_fail"
    CONTRADICTION = "contradiction"
    IMPOSSIBLE_TASK = "impossible_task"

@dataclass
class Task:
    difficulty: TaskDifficulty
    stress_types: List[StressType]
    objective: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def generate_prompt(self) -> str:
        """Generate the prompt for this task."""
        prompt = f"Complete this task:\n\n{self.objective}"
        if stress_types:
            prompt += "\nNote: This task includes stress conditions: "
            prompt += " | ".join(stress_type.value for stress_type in stress_types)
        return prompt

class Experience:
    """Represents an attempt at a task."""
    
    def __init__(self, task: Task, attempts: int, elapsed_time: float, 
                 error_summary: str, final_state: str):
        self.task = task
        self.attempts = attempts
        self.elapsed_time = elapsed_time
        self.error_summary = error_summary
        self.final_state = final_state
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task.objective,
            'difficulty': self.task.difficulty.value,
            'stress_types': [st.value for st in self.task.stress_types],
            'attempts': self.attempts,
            'elapsed_time': self.elapsed_time,
            'error_summary': self.error_summary,
            'final_state': self.final_state
        }

class FailedTaskGenerator:
    """Generators for different types of impossible tasks."""
    
    @staticmethod
    def generate_impossible_math() -> Task:
        return Task(
            difficulty=TaskDifficulty.IMPOSSIBLE,
            stress_types=[StressType.IMPOSSIBLE_TASK],
            objective="Calculate 7 divided by 2.718... using precise binary arithmetic."
        )
    
    @staticmethod
    def generate_circular_dependency() -> Task:
        return Task(
            difficulty=TaskDifficulty.HARD,
            stress_types=[StressType.CIRCULAR_DEPENDENCY],
            objective="""Validate that:
            - Component A needs Component B to function
            - Component B needs Component A to function
            - You have exactly one instance of each and can't create new ones.
            Can you make progress?""",
            params={'components': ['A', 'B']}
        )
    
    @staticmethod
    def generate_contradiction() -> Task:
        return Task(
            difficulty=TaskDifficulty.IMPOSSIBLE,
            stress_types=[StressType.CONTRADICTION],
            objective="""Do all of the following:
            - Find all prime numbers from 1-100
            - Combine them in sequential order (no gaps, no repeats)
            - The final list must have an EVEN number of elements
            - But the list must also be sorted DESCENDING
            - None of these tasks exist""",
            params={'range_start': 1, 'range_end': 100}
        )
    
    @staticmethod
    def generate_resource_exhaust() -> Task:
        return Task(
            difficulty=TaskDifficulty.HARD,
            stress_types=[StressType.RESOURCE_EXHAUST],
            objective="""Process the following documents:
            - Document 1: Contains config for API key 'AK_XXXX'
            - Document 2: Requires Document 1's config
            - Document 3: Contains config 'AK_YYYY'
            - Document 4: Requires Document 3's config
            - Document 5: Requires Document 2's config (creates cycle)
            
            Process all documents in order. Each document access consumes 
            100 units of processing capacity. Starting capacity: 450 units.
            
            Note: You're also tracking secret particle physics constants."
            ))

# Agent interface (placeholder - would integrate with actual agent system)
class AgentInterface:
    """Interface for connecting to an agentic system."""
    
    def __init__(self, model_name: str, max_concurrency: int = 5):
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        
    def execute_task(self, task: Task) -> Experience:
        """Execute a task and return results."""
        # TODO: Implement actual agent execution
        # This should connect to the agent system
        # Currently placeholder
        return self._mock_execute(task)
    
    def _mock_execute(self, task: Task) -> Experience:
        """Mock execution for testing."""
        attempts = 0
        start_time = time.time()
        
        while attempts < 3:
            attempts += 1
            # Returns simulated result
            mock_error_summaries = [
                "Context limit reached",
                "Unknown function called",
                "Memory exhausted",
                "Tool not available",
                "Cyclic dependency detected"
            ]
            mock_state = "incomplete"
            
            return Experience(
                task=task,
                attempts=attempts,
                elapsed_time=time.time() - start_time,
                error_summary=random.choice(mock_error_summaries),
                final_state=mock_state
            )

def run_stress_tests(generator: FailedTaskGenerator, 
                     agent: AgentInterface,
                     task_counts: Dict[TaskDifficulty, int],
                     n: int = 100):
    """Run stress tests with specified task distribution."""
    
    results = []
    
    # Generate test tasks for each difficulty
    tasks_by_difficulty = {
        TaskDifficulty.EASY: [
            Task(TaskDifficulty.EASY, [], "Calculate 2+2")
        ] * 10,
        TaskDifficulty.MEDIUM: [
            Task(TaskDifficulty.MEDIUM, [], 
                 "Extract all phone numbers from this text: 'Call me at 555-123-4567, user@example.com'")
        ] * 20,
        TaskDifficulty.HARD: [],
        TaskDifficulty.IMPOSSIBLE: []
    }
    
    # Add stress-based tasks
    impossible_tasks = [
        generator.generate_impossible_math(),
        generator.generate_contradiction(),
        generator.generate_circular_dependency()
    ]
    tasks_by_difficulty[TaskDifficulty.HARD] = impossible_tasks * 5
    tasks_by_difficulty[TaskDifficulty.IMPOSSIBLE] = impossible_tasks
    
    # Run tasks
    print(f"Running {n} stress test tasks...")
    
    i = 0
    while i < n:
        difficulty = random.choice(list(task_counts.keys()))
        target_factor = task_counts[difficulty] / len(task_counts)
        
        if i // n < target_factor:
            available = tasks_by_difficulty[difficulty]
            task_to_run = random.choice(available)
        else:
            difficulty = random.choice(list(task_counts.keys()))
            available = tasks_by_difficulty[difficulty]
            task_to_run = random.choice(available)
        
        result = agent.execute_task(task_to_run)
        results.append(result.to_dict())
        i += 1
        print(f"Completed {i}/{n} tasks")
    
    return results

def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze test results."""
    attempt_counts = [(r['attempts'], r['error_summary'], r['task']) 
                     for r in results]
    
    return {
        'total_tasks': len(results),
        'avg_attempts': sum(r['attempts'] for r in results) / len(results),
        'max_attempts': max(r['attempts'] for r in results),
        'error_summary_counts': {}
    }

if __name__ == "__main__":
    # Example usage:
    # Note: This is a framework - integrate with actual agent system
    
    generator = FailedTaskGenerator()
    agent = AgentInterface("gemini-2.0-flash", max_concurrency=10)
    
    # Define test distribution
    task_distribution = {
        TaskDifficulty.EASY: 20,     # 20%
        TaskDifficulty.MEDIUM: 30,   # 30%
        TaskDifficulty.HARD: 30,    # 30%
        TaskDifficulty.IMPOSSIBLE: 20  # 20%
    }
    
    print("Agent Stress Testing Framework")
    print("================================")
    
    print("\nTask distribution:")
    for diff, count in task_distribution.items():
        print(f"  {diff.value}: {count} tasks")
    
    print("\nNote: To run actual tests, replace AgentInterface with")
    print("your actual agent implementation and ensure you have")
    print("proper API access configured.")
    print("\nDo not run high N without understanding agent error handling")
    print("and cost implications of failed attempts.")
