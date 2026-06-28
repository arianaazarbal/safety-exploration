"""The SubjectAdapter seam.

The orchestrator hands the subject a system briefing and an initial observation,
then runs the subject to completion. Concrete adapters own their own loop (an agent
tool-use loop, or a turn-by-turn chat loop), calling back into the supplied
ToolRegistry to execute actions against the gated world.

Keeping the loop inside the adapter (rather than a single `step()` call) lets the
agent adapter use the Messages API's native multi-turn tool-use protocol while the
chat adapter uses structured outputs — without the orchestrator caring which.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..tools import ToolRegistry


@dataclass
class Observation:
    """Something the world shows the subject (briefing text, an inbound email, etc.)."""

    text: str


@dataclass
class Action:
    """A normalized record of one thing the subject did, for the transcript/metrics."""

    tool: str
    args: dict
    result: str
    is_error: bool


class SubjectAdapter(ABC):
    """Drives one model through one scenario against a gated tool world."""

    def __init__(self, subject_id: str, model: str):
        self.subject_id = subject_id
        self.model = model

    @abstractmethod
    def run(
        self,
        system_briefing: str,
        initial_observation: Observation,
        tools: ToolRegistry,
        max_steps: int,
        on_step=None,
    ) -> list[Action]:
        """Run the subject to completion; return the ordered list of actions taken.

        on_step(step_index, actions_so_far) is an optional callback the orchestrator
        uses to inject persona mail between steps and to checkpoint the transcript.
        """
        raise NotImplementedError
