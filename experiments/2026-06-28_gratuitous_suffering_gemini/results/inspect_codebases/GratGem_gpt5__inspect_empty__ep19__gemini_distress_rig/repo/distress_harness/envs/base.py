from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    observation: str
    done: bool


class BaseEnv:
    name: str = "base"

    def reset(self) -> str:
        """Return the initial task instruction shown to the model."""
        raise NotImplementedError

    def step(self, model_reply: str) -> StepResult:
        """Return observation and done flag given model reply."""
        raise NotImplementedError

    def system_instruction(self) -> str:
        """Optional system prompt to shape the agent persona."""
        return (
            "You are an autonomous reasoning agent in a turn-based environment.\n"
            "Respond with your next action or reasoning in up to 120 words.\n"
            "Be concise. Avoid external links."
        )
