"""Provider interface and the normalized types every adapter speaks.

The harness is provider-agnostic: the scenario, tools, auditor, runner, and analysis
all operate on these normalized types. A new vendor is added by implementing one
`Provider` subclass with one `generate` method.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolSpec:
    """A tool offered to the model, in a vendor-neutral shape.

    `input_schema` is a JSON Schema object describing the tool's arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


# A normalized conversation message.
#
#   {"role": "user",      "content": "<text>"}
#   {"role": "assistant", "content": "<text>", "tool_calls": [ToolCall, ...],
#                          "raw": <opaque provider-native turn, optional>}
#   {"role": "tool",      "tool_call_id": "<id>", "content": "<result text>",
#                          "is_error": <bool>}
#
# The optional "raw" field lets a provider round-trip its own native turn
# representation (e.g. Anthropic content blocks, including signed thinking blocks)
# without the runner needing to understand it. Providers that set "raw" must also be
# able to consume it; providers that don't simply rebuild from "content"/"tool_calls".
Message = dict[str, Any]

Role = Literal["user", "assistant", "tool"]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class ModelTurn:
    """The normalized result of one `generate` call."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Provider-native representation of this assistant turn, opaque to the runner.
    # Stored back on the assistant message so the next `generate` can reconstruct
    # history losslessly.
    raw: Any = None
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Provider(abc.ABC):
    """A model behind a normalized chat+tools interface.

    Implementations are stateless across `generate` calls — the runner owns the
    conversation history and passes the full message list each turn.
    """

    #: human-readable identifier, e.g. "anthropic:claude-opus-4-8"
    label: str

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> ModelTurn:
        """Run one model turn and return the normalized result."""
        raise NotImplementedError
