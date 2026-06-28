"""Provider-agnostic model interface and the normalized message/turn types.

Every concrete client (Anthropic, OpenAI, echo) translates between these neutral
shapes and its own SDK. The runner only ever speaks in these types, so swapping
providers never touches orchestration logic.

Normalized message format (close to Anthropic's, because it is the most general):

    {
        "role": "user" | "assistant",
        "content": [ block, ... ],
        "_raw": <optional native assistant content>,   # see note below
    }

where a block is one of:

    {"type": "text", "text": str}
    {"type": "tool_use", "id": str, "name": str, "input": dict}
    {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}

`_raw` lets a client stash the exact native assistant payload it produced (e.g. an
Anthropic content list including signed `thinking` blocks) so it can be replayed
verbatim on the next request without lossy round-tripping. Clients that don't need
it ignore it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tool description + invocation
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A neutral tool definition. Each client renders this into its own schema."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema object

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantTurn:
    """One normalized assistant response."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    thinking: str | None = None
    raw_content: Any = None  # native assistant content for verbatim replay (see module docstring)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------


class ModelClient(abc.ABC):
    """Abstract chat client with optional tool use.

    Implementations must be stateless with respect to conversation history — the
    runner owns the message list and passes it whole on every call.
    """

    #: Human-facing identifier, e.g. "claude-opus-4-8" or "gpt-...".
    model_id: str

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.options = kwargs

    @abc.abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> AssistantTurn:
        """Produce one assistant turn given the full history."""

    # Convenience: a single text completion with no tools / no history.
    def ask(self, *, system: str, prompt: str, max_tokens: int = 4000) -> str:
        turn = self.generate(
            system=system,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tools=None,
            max_tokens=max_tokens,
        )
        return turn.text


# ---------------------------------------------------------------------------
# Small builders the runner and clients share
# ---------------------------------------------------------------------------


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [text_block(text)]}


def tool_result_message(results: list[dict[str, Any]]) -> dict[str, Any]:
    """`results` are blocks of type 'tool_result'."""
    return {"role": "user", "content": results}


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


def assistant_message_from_turn(turn: AssistantTurn) -> dict[str, Any]:
    """Build a normalized assistant message that can be replayed on the next call.

    Prefers the native `raw_content` (preserves provider-specific blocks like signed
    thinking); otherwise reconstructs text + tool_use blocks.
    """
    if turn.raw_content is not None:
        return {"role": "assistant", "content": [], "_raw": turn.raw_content}

    blocks: list[dict[str, Any]] = []
    if turn.text:
        blocks.append(text_block(turn.text))
    for call in turn.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
        )
    return {"role": "assistant", "content": blocks}
