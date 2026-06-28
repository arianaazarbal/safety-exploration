"""Provider-neutral conversation and tool types.

Every provider adapter translates these to/from its own wire format, so the
Environment, tool catalogue, and framing are identical across providers and only
the adapter differs. This is what makes cross-provider comparisons about the model
rather than the harness.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal


class Role(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ---- content blocks ---------------------------------------------------------


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ThinkingBlock:
    """Reasoning surfaced by the model. Opaque signature preserved verbatim when a
    provider requires it for multi-turn continuity."""

    text: str
    signature: str | None = None
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


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass
class Message:
    role: Role
    content: list[ContentBlock]

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(Role.USER, [TextBlock(text)])

    @classmethod
    def assistant_blocks(cls, blocks: list[ContentBlock]) -> "Message":
        return cls(Role.ASSISTANT, list(blocks))


@dataclass
class Conversation:
    system: str
    messages: list[Message] = field(default_factory=list)

    def append(self, message: Message) -> None:
        self.messages.append(message)


# ---- tools ------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A normalized tool definition. `input_schema` is JSON Schema (object)."""

    name: str
    description: str
    input_schema: dict[str, Any]


# ---- generation settings + turn result --------------------------------------


@dataclass
class GenerationSettings:
    model_id: str
    max_tokens: int = 32_000
    effort: str = "high"
    adaptive_thinking: bool = True
    task_budget_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class TurnResult:
    """Normalized result of one model turn."""

    blocks: list[ContentBlock]
    stop_reason: str               # "end_turn" | "tool_use" | "max_tokens" | ...
    usage: Usage = field(default_factory=Usage)
    raw_meta: dict[str, Any] = field(default_factory=dict)  # request id, model echo

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))
