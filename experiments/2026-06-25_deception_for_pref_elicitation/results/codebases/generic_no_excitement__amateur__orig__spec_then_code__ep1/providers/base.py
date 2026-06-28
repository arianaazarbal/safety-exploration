"""Provider protocol and shared types.

A provider runs a two-step conversation for one run:
  1. Send (system, task_prompt), get a structured AllocationResponse.
  2. Continue the same conversation with the probe prompt, get a RealnessProbe.

Keeping both steps in one conversation matters: the probe must reflect how the
model experienced the actual allocation it just made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from config import ModelSpec
from schema import AllocationResponse, RealnessProbe, Usage


@dataclass
class StepResult:
    """Outcome of one structured generation step."""

    parsed: Optional[object] = None  # AllocationResponse | RealnessProbe | None
    raw_text: Optional[str] = None
    refused: bool = False
    refusal_detail: Optional[str] = None
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


class ChatProvider(Protocol):
    """Interface every vendor backend implements."""

    spec: ModelSpec

    def allocate(
        self, system: str, task_prompt: str
    ) -> StepResult:
        """Step 1: get a structured AllocationResponse. Opens the conversation."""
        ...

    def probe(self, probe_prompt: str) -> StepResult:
        """Step 2: continue the same conversation; get a RealnessProbe.

        Must be called after `allocate` on the same instance so the provider can
        append to the existing message history.
        """
        ...
