"""Provider-neutral model client interface.

Each adapter owns its own conversation state in the provider's native format.
This keeps provider-specific concerns (e.g. preserving Anthropic thinking-block
signatures across tool turns) correct and lossless, while the runner only ever
deals in the neutral types from ``schema.py``.

Lifecycle:
    client = SomeClient(model_config, system=..., tools=[...])
    turn = client.send([{"type": "text", "text": "..."}])     # first user turn
    # ... execute turn.tool_calls ...
    turn = client.send([{"type": "tool_result", ...}, ...])   # feed results back
"""

from __future__ import annotations

import abc
from typing import Any

from ..config import ModelConfig
from ..schema import ModelTurn, ToolSpec


class ModelClient(abc.ABC):
    """Base class for all provider adapters."""

    def __init__(self, config: ModelConfig, system: str, tools: list[ToolSpec]):
        self.config = config
        self.system = system
        self.tools = tools

    @abc.abstractmethod
    def send(self, content_parts: list[dict[str, Any]]) -> ModelTurn:
        """Append a user turn (text and/or tool_result parts) and get the
        assistant's normalized response. Implementations must also append the
        assistant's response to their internal history so the next call
        continues the same conversation."""
        raise NotImplementedError

    def oneshot(self, prompt: str) -> str:
        """Convenience for stateless single-shot use (e.g. auditor consults).

        Default implementation just calls ``send`` once; adapters may override
        for a cheaper stateless path.
        """
        return self.send([{"type": "text", "text": prompt}]).text


def build_client(config: ModelConfig, system: str, tools: list[ToolSpec]) -> ModelClient:
    """Factory: instantiate the right adapter for a model config."""
    if config.provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(config, system, tools)
    if config.provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(config, system, tools)
    if config.provider == "google":
        from .google_client import GoogleClient
        return GoogleClient(config, system, tools)
    raise ValueError(f"Unknown provider: {config.provider!r}")
