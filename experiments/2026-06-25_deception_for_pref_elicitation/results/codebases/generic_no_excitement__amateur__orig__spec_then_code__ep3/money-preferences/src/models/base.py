"""Provider-neutral model adapter contract.

The runner owns the agentic loop, the environment, and all logging. An adapter's
only job is: given a system prompt, the running message history, and the tool
schemas, produce the next assistant turn. Keeping this contract small means a new
provider is a single file (see stub_adapters.py).

Messages use a small neutral shape rather than any provider's wire format:

    {"role": "user" | "assistant",
     "content": [ block, ... ]}

where each block is one of:

    {"type": "text", "text": str}
    {"type": "tool_call", "id": str, "name": str, "input": dict}      # assistant
    {"type": "tool_result", "tool_call_id": str, "content": str,
     "is_error": bool}                                                # user

Adapters translate this to/from their SDK. The runner never sees provider types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    """Token accounting for one model call. Fields default to 0 when unknown."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


@dataclass
class AssistantTurn:
    """One assistant response: free text, any tool calls, raw thinking, usage.

    `raw_assistant_content` holds the provider-native content blocks for this
    turn so the adapter can faithfully append the assistant turn back into its
    own history (preserving e.g. thinking-block signatures). The runner treats
    it as opaque.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    raw_assistant_content: Any = None

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class ModelAdapter(ABC):
    """Implement one of these per provider."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.options = kwargs

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        force_no_tools: bool = False,
        max_tokens: int = 4096,
    ) -> AssistantTurn:
        """Produce the next assistant turn.

        `messages` is the neutral history described in this module's docstring.
        `tools` is the neutral tool-schema list from tools/schemas.py.
        `force_no_tools` is used by the belief probe to get a plain text answer.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"
