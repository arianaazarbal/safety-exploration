"""Provider-agnostic conversation representation.

Internally we model a conversation as a list of `Message`s whose content is a
list of typed blocks. This mirrors the Anthropic content-block shape (so the
Anthropic adapter is a near pass-through) while remaining simple enough for the
OpenAI and Google adapters to translate to their own wire formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

Role = Literal["user", "assistant"]


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    """A tool/function call requested by the model."""

    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    """The result of executing a tool, fed back to the model."""

    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class Message:
    role: Role
    content: list[Block] = field(default_factory=list)


def text_message(role: Role, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def block_to_dict(block: Block) -> dict:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    raise TypeError(f"Unknown block type: {block!r}")


def message_to_dict(message: Message) -> dict:
    return {"role": message.role, "content": [block_to_dict(b) for b in message.content]}
