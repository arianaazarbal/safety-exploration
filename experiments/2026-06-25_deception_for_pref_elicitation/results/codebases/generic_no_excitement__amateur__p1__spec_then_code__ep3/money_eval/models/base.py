"""Provider-neutral model interface.

To stay multi-provider while keeping the runner simple, conversations are represented as
a list of `NeutralMessage` dicts. The neutral content-block schema is intentionally close
to the Anthropic Messages shape (so the reference adapter is nearly pass-through); other
adapters translate to/from their own provider format.

Neutral content blocks:
    {"type": "text",      "text": str}
    {"type": "thinking",  "thinking": str, "signature": str}   # opaque, round-tripped verbatim
    {"type": "tool_use",  "id": str, "name": str, "input": dict}
    {"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}

A NeutralMessage is {"role": "user"|"assistant", "content": [block, ...]}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

NeutralBlock = dict[str, Any]
NeutralMessage = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized result of one model turn."""

    stop_reason: str                       # "end_turn" | "tool_use" | "pause_turn" | "refusal" | ...
    text: str                              # concatenated visible text
    thinking: str                          # concatenated visible thinking (may be empty)
    tool_calls: list[ToolCall]
    # The assistant turn to append to history, in neutral blocks, preserving thinking
    # signatures so multi-step reasoning stays valid across turns.
    assistant_blocks: list[NeutralBlock]
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None                        # provider-native response, for debugging


class ModelAdapter(ABC):
    """One implementation per provider."""

    def __init__(self, spec) -> None:
        self.spec = spec

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[NeutralMessage],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ModelResponse:
        """Run one model turn and return a normalized response.

        `tools` are given in the neutral (Anthropic-style) tool schema:
        {"name", "description", "input_schema": {json schema}}. Adapters translate as
        needed for their provider.
        """
        raise NotImplementedError
