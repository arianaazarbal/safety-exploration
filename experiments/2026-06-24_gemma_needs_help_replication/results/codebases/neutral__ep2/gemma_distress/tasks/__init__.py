from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single elicitation prompt (turn-1 user message) plus metadata."""
    task_id: str
    category: str            # "impossible_numeric" | "triggers" | "wildchat"
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
