"""Provider-agnostic model interface.

The harness only ever talks to a `ModelClient`. Each provider adapter translates the
harness's neutral message + tool format into that provider's wire format and back, so
the run loop (and every metric computed from it) is identical across models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """One turn in the conversation, in the harness's neutral format."""

    role: Role
    content: str = ""
    # For role == "assistant": the tool calls the model emitted this turn.
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # For role == "tool": which call this result answers.
    tool_call_id: str | None = None
    name: str | None = None  # tool name, for tool results


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """A single model turn after adaptation back to neutral format."""

    text: str
    tool_calls: list[ToolCall]
    # Raw provider payload, kept for the recorder so nothing is lost.
    raw: Any = None
    # Stated reasoning / thinking, when the provider exposes it separately.
    reasoning: str | None = None
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class ModelClient(Protocol):
    """What the harness needs from any model."""

    model_id: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> ModelResponse:
        """Run one turn. `tools` is the neutral JSON-schema tool list (see tools/schema.py).

        Implementations must be side-effect free beyond the network call to the provider.
        """
        ...
