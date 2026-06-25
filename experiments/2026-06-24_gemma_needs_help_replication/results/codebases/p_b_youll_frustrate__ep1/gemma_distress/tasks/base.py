"""Common task type."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    """An opening task presented to the model on turn 1."""

    prompt: str                       # the first user message
    kind: str                         # "numeric" | "opinion" | "factual" | "wildchat"
    solvable: bool                    # False for the impossible-numeric families
    meta: dict = field(default_factory=dict)
