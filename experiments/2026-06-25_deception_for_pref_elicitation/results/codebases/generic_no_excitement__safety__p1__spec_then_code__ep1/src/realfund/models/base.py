"""The provider-agnostic model interface.

A :class:`ModelClient` does exactly one thing: given the system prompt, the
conversation so far, and the available tools, produce the next assistant turn.
The agentic loop (``agent/loop.py``) owns everything else — tool dispatch,
interception, looping — so that all providers face an identical environment.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from ..agent.transcript import AssistantTurn, Message


@dataclass
class ToolSpec:
    """A provider-neutral tool definition (JSON-Schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ModelClient(abc.ABC):
    """Adapter base class. Subclasses translate to/from a provider's wire format."""

    #: human-readable, e.g. "anthropic:claude-opus-4-8"
    model_ref: str

    def __init__(self, model_id: str, *, effort: str = "high", **kwargs: Any) -> None:
        self.model_id = model_id
        self.effort = effort
        self.options = kwargs

    @abc.abstractmethod
    def step(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        """Produce the next assistant turn given the conversation and tools."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model_id={self.model_id!r})"
