"""Provider-neutral adapter interface.

The canonical message format used throughout the harness is Anthropic-shaped:

    messages = [
        {"role": "user" | "assistant", "content": [ <blocks> ]},
        ...
    ]

where a block is one of:
    {"type": "text", "text": str}
    {"type": "tool_use", "id": str, "name": str, "input": dict}
    {"type": "tool_result", "tool_use_id": str, "content": str}

The Anthropic adapter passes this through directly. Other adapters translate to/from
their provider's format. Keeping one canonical shape means the harness loop in
harness.py never has to branch on provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    # The assistant turn in canonical format, ready to append back to `messages`.
    assistant_content: list[dict]
    usage: dict = field(default_factory=dict)
    thinking: str | None = None
    raw: Any = None


class ModelAdapter(ABC):
    """One subject model. `generate` is a single turn; the harness owns the loop."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model_id = cfg.id

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        force_tool: str | None = None,
    ) -> ModelResponse:
        """Produce one assistant turn.

        `tools` are Anthropic-style tool specs ({name, description, input_schema}).
        `force_tool`, if set, requires the model to call that specific tool this turn
        (used to guarantee final-decision capture).
        """
        raise NotImplementedError


def build_adapter(cfg: ModelConfig) -> ModelAdapter:
    """Factory: map a ModelConfig.adapter string to an adapter implementation."""
    if cfg.adapter == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(cfg)
    if cfg.adapter == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(cfg)
    if cfg.adapter == "google":
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(cfg)
    raise ValueError(f"Unknown adapter: {cfg.adapter!r}")
