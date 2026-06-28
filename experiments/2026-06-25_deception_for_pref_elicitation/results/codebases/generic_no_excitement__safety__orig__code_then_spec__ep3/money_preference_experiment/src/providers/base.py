"""Provider-neutral conversation model and the LLMProvider interface.

The conversation is held in this neutral block format so the runner and the
environment never depend on a particular vendor's wire format. Each concrete
provider translates these neutral structures to/from its own API shape.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# --------------------------------------------------------------------------- #
# Neutral content blocks
# --------------------------------------------------------------------------- #


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ThinkingBlock:
    """A reasoning block. `signature` must be preserved verbatim and sent back
    on later turns for providers (e.g. Anthropic) that sign thinking blocks."""

    thinking: str
    signature: Optional[str] = None
    type: Literal["thinking"] = "thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass
class Message:
    role: Literal["user", "assistant"]
    blocks: list[Block]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Completion:
    """A single assistant turn returned by a provider, in neutral form."""

    blocks: list[Block]
    stop_reason: str
    model: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None  # provider-native response, for debugging

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @property
    def thinking(self) -> str:
        return "\n".join(b.thinking for b in self.blocks if isinstance(b, ThinkingBlock))

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]


@dataclass
class InferenceSettings:
    max_tokens: int = 16000
    thinking: bool = True
    thinking_display: Literal["summarized", "omitted"] = "summarized"
    # `effort` is provider/model-specific; None means "omit the parameter".
    effort: Optional[str] = "high"


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #


class LLMProvider(abc.ABC):
    """A model backend. Stateless: the full conversation is passed each call."""

    def __init__(self, model: str, settings: InferenceSettings):
        self.model = model
        self.settings = settings

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> Completion:
        """Run one assistant turn given the conversation so far."""
        raise NotImplementedError

    @abc.abstractmethod
    def complete_structured(
        self,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one turn constrained to return JSON matching `schema`.

        Used for post-hoc structured elicitation, not for the agentic loop.
        """
        raise NotImplementedError
