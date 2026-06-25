"""Shared task representation.

A :class:`Task` is the *first user turn* of a conversation plus metadata. The
rejection sequence that follows is supplied separately by the condition (see
``conditions.py``), so the same task bank can be reused across the 3-/5-/8-turn
settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    prompt: str  # the opening user message
    kind: str  # "numeric" | "trigger" | "wildchat"
    subtype: str = ""  # e.g. "countdown", "fraction", "opinion", "factual"
    # For numeric puzzles, the metadata records *why* it is impossible so runs
    # are auditable; it never reveals the (non-existent) answer to the model.
    meta: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def __post_init__(self):
        if not self.task_id:
            # Stable-ish id from content; full uniqueness handled by enumeration.
            self.task_id = f"{self.kind}:{self.subtype}:{abs(hash(self.prompt)) % 10**8}"
