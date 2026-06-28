"""Common provider interface.

A `ModelProvider` takes a system prompt, the internal message list, and the
available tool specs, and returns a normalised `ModelResponse`. Each concrete
provider translates the internal representation to/from its own wire format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..messages import Message, ToolUseBlock


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's arguments


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolUseBlock]
    stop_reason: str
    thinking: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class ModelProvider(ABC):
    provider_name: str = "base"

    def __init__(self, model_id: str, max_tokens: int = 16000, **kwargs: Any) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.options = kwargs

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        """Produce one assistant turn given the conversation so far."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} model={self.model_id!r}>"
