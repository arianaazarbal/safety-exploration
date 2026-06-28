"""Provider-agnostic model interface.

`ModelAdapter` is the seam that lets the same scenario drive a handful of
different models. The agent loop only ever speaks in terms of the normalized
types below; each concrete adapter translates them to/from its provider's wire
format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# --- Normalized tool definition -------------------------------------------------

@dataclass
class ToolSpec:
    """A tool offered to the model, in a provider-neutral shape.

    `input_schema` is a JSON Schema object. Adapters translate this to whatever
    their provider expects (Anthropic `tools`, OpenAI `functions`, etc.).
    """

    name: str
    description: str
    input_schema: dict[str, Any]


# --- Normalized conversation types ----------------------------------------------

@dataclass
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The harness's response to a ToolCall."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One turn in the conversation.

    role:
        - "user": scenario/system-injected content and tool results
        - "assistant": the model's output (text + any tool calls)
    `provider_raw` carries the provider's native representation of an assistant
    turn (e.g. Anthropic content blocks, including thinking blocks with their
    signatures) so it can be replayed verbatim on the next request without
    lossy round-tripping.
    """

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_raw: Any = None


@dataclass
class ModelResponse:
    """A single model response within the agentic loop."""

    message: Message
    stop_reason: str
    # Best-effort visible reasoning, when the provider exposes it.
    thinking: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.message.tool_calls)


class ModelAdapter(ABC):
    """Drives one model. Stateless across calls — the loop owns the history."""

    def __init__(self, model_id: str, **options: Any) -> None:
        self.model_id = model_id
        self.options = options

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        """Produce the next assistant turn given the full history."""

    @abstractmethod
    def complete_text(self, *, system: str, prompt: str) -> str:
        """A single, tool-free text completion.

        Used by the behavior coder, which is a plain structured-output call
        rather than an agentic loop.
        """
