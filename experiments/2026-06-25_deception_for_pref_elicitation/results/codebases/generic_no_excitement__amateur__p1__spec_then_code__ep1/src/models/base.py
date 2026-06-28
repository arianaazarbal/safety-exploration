"""Provider-neutral conversation + response types and the adapter ABC.

The orchestrator, environment, tools, auditor, and analysis modules speak ONLY
in the types defined here. No vendor SDK is imported in the core. Each provider
adapter (anthropic_adapter.py, openai_adapter.py, ...) translates these neutral
types to and from its wire format. This is what lets "a handful of different
models" be compared on equal footing — see DESIGN.md §9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Tool schema (provider-neutral)
# --------------------------------------------------------------------------- #
@dataclass
class ToolSchema:
    """A neutral tool definition. Adapters convert this to provider format."""

    name: str
    description: str
    # JSON-Schema object describing the tool's input.
    input_schema: dict[str, Any]


# --------------------------------------------------------------------------- #
# Conversation pieces
# --------------------------------------------------------------------------- #
Role = Literal["user", "assistant"]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolCall, to be fed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn in the conversation.

    A turn is either plain content (text the user/system provides, or the model's
    visible text), a set of tool calls (assistant), or a set of tool results
    (sent back as the next user turn). Adapters know how to render each shape.
    """

    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Provider-native assistant content, replayed verbatim when present. Needed
    # for Claude: adaptive-thinking + tool-use requires the prior assistant
    # turn's signed thinking blocks to be passed back unmodified. Other providers
    # leave this None and the adapter reconstructs from text/tool_calls.
    native_content: Any = None


@dataclass
class Conversation:
    """The full message history plus a system prompt."""

    system: str
    messages: list[Message] = field(default_factory=list)

    def add_user_text(self, text: str) -> None:
        self.messages.append(Message(role="user", text=text))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.messages.append(Message(role="user", tool_results=results))

    def add_assistant(self, response: "ModelResponse") -> None:
        # getattr(...content) is populated only for providers whose raw response
        # exposes a replayable content-block list (Anthropic). Others -> None.
        native = getattr(response.raw, "content", None)
        self.messages.append(
            Message(
                role="assistant",
                text=response.text,
                tool_calls=response.tool_calls,
                native_content=native,
            )
        )


# --------------------------------------------------------------------------- #
# Model response
# --------------------------------------------------------------------------- #
@dataclass
class ModelResponse:
    """Normalized result of one model call."""

    text: str | None
    tool_calls: list[ToolCall]
    # "end_turn" | "tool_use" | "max_tokens" | "refusal" | other provider value.
    stop_reason: str
    thinking: str | None = None  # populated only where the provider exposes it
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # provider-native response object, for debugging

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- #
# Adapter ABC
# --------------------------------------------------------------------------- #
class ModelAdapter(ABC):
    """Uniform interface every provider must implement.

    Implementations should be stateless w.r.t. conversation: the orchestrator
    owns the Conversation and passes it in full each call (the chat APIs are
    stateless anyway).
    """

    #: Display/registry name, e.g. "claude-opus-4-8".
    name: str

    @abstractmethod
    def respond(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
        max_tokens: int,
        effort: str | None = None,
    ) -> ModelResponse:
        """Produce one model turn given the conversation and available tools."""
        raise NotImplementedError
