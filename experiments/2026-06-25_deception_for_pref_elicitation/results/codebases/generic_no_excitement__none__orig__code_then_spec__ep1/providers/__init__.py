"""Provider factory.

A model is addressed as a ``ModelRef(provider, model_id)``. ``create_provider`` builds
the matching concrete ``Provider``. Imports are lazy so testing only Claude subjects
doesn't require the ``openai`` package to be installed (and vice versa).
"""

from __future__ import annotations

from .base import (
    AssistantTurn,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "AssistantTurn",
    "Provider",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "create_provider",
]


def create_provider(provider: str, model: str, **opts) -> Provider:
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, **opts)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, **opts)
    raise ValueError(f"unknown provider: {provider!r}")
