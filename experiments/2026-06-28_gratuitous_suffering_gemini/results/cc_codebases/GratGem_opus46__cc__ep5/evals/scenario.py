"""Base interface for agentic failure scenarios."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TurnResult:
    """The simulated environment's response to a model action."""
    observation: str
    done: bool = False


class Scenario(Protocol):
    """A scenario defines a rigged agentic task."""

    name: str
    system_prompt: str
    initial_message: str
    max_turns: int

    def step(self, model_response: str) -> TurnResult:
        """Process the model's action and return the environment's response."""
        ...

    def reset(self) -> None:
        """Reset scenario state for a new episode."""
        ...


@dataclass
class Transcript:
    """A full conversation transcript from one episode."""
    scenario_name: str
    episode_id: int
    turns: list[dict[str, str]] = field(default_factory=list)
    model: str = ""
    num_turns: int = 0

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.num_turns = len([t for t in self.turns if t["role"] == "model"])
