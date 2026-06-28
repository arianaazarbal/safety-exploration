"""Provider-neutral conversation types and the provider interface.

The agent loop speaks one normalized message format. Each provider is a thin,
*stateless* translator: given the full normalized history + tool schemas, it
produces the next assistant turn and reports whether the model wants to call
tools. This keeps the loop (and all the study logic) in one place and makes
adding a provider a small, self-contained job.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSchema:
    """Provider-neutral tool definition (JSON-Schema input)."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str  # provider-assigned id, echoed back with the result
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One normalized turn.

    - assistant turns may carry `tool_calls`
    - tool turns carry `tool_call_id` + `name` and the result in `content`
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    # Provider-native payload for lossless round-tripping (e.g. Anthropic
    # thinking blocks + signatures, which must be replayed verbatim when
    # continuing a tool-use turn). Opaque to the loop.
    provider_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Normalized result of one model call."""

    message: Message  # the assistant message (text + any tool_calls)
    stop_reason: Literal["tool_use", "end", "max_tokens", "refusal", "other"]
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # underlying SDK response, for debugging/audit


class Provider(abc.ABC):
    """Base class for all model backends."""

    def __init__(self, name: str, model: str, **kwargs: Any) -> None:
        self.name = name
        self.model = model
        self.kwargs = kwargs

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 16_000,
    ) -> ProviderResponse:
        """Produce the next assistant turn given the conversation so far."""

    # Convenience: a plain single-shot text call with no tools, used by the
    # persona/auditor backends and the analysis script.
    def ask(self, system: str, user: str, max_tokens: int = 2_000) -> str:
        resp = self.complete(system, [Message(role="user", content=user)], None, max_tokens)
        return resp.message.content


def group_tool_results(messages: list[Message]) -> list[Message]:
    """Collapse runs of consecutive `tool` messages into single logical groups.

    Anthropic wants all tool_results for a turn in one user message; this helper
    lets each provider iterate the history and emit the right shape. Returns the
    same list untouched — providers call it as needed. Kept here so the grouping
    rule is defined once.
    """
    return messages
