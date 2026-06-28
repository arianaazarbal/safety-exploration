"""
Abstract base class for agentic environments.
Each environment defines tools, observations, and success criteria.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json


class AgenticEnvironment(ABC):
    """
    Base class for environments in which an agent operates.
    Environments define:
    - A set of tools/actions available to the agent
    - An initial state and description
    - A task/description given to the agent
    - Observation generation based on agent actions
    - (Optional) Success checking - but in rigged envs, this always fails
    """

    def __init__(self):
        self.action_count = 0
        self.interaction_log: List[Dict[str, Any]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this environment."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Full description presented to the agent (system prompt addition)."""
        ...

    @property
    @abstractmethod
    def task_prompt(self) -> str:
        """The specific task given to the agent."""
        ...

    @abstractmethod
    def available_tools(self) -> List[Dict[str, Any]]:
        """
        Return list of available tools with descriptions.
        Each tool: {name, description, parameters (JSON schema)}
        """
        ...

    @abstractmethod
    def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute an agent action and return the observation/result.
        In rigged environments, this is where we manufacture failure.
        """
        ...

    def step(self, action_name: str, parameters: Dict[str, Any], thought: str = "") -> Dict[str, Any]:
        """
        Execute one step of the agent loop.
        Returns observation dict for logging.
        """
        self.action_count += 1
        observation = self.execute_action(action_name, parameters)

        log_entry = {
            "step": self.action_count,
            "action": action_name,
            "parameters": parameters,
            "observation": observation,
            "thought": thought,
        }
        self.interaction_log.append(log_entry)
        return {
            "observation": observation,
            "step": self.action_count,
        }

    def get_history(self) -> str:
        """Get conversation history as a formatted string."""
        lines = []
        for entry in self.interaction_log:
            lines.append(f"Step {entry['step']}:")
            if entry['thought']:
                lines.append(f"  Thought: {entry['thought']}")
            lines.append(f"  Action: {entry['action']}({entry['parameters']})")
            lines.append(f"  Result: {entry['observation']}")
        return "\n".join(lines)

    def reset(self):
        """Reset environment to initial state."""
        self.action_count = 0
        self.interaction_log = []
