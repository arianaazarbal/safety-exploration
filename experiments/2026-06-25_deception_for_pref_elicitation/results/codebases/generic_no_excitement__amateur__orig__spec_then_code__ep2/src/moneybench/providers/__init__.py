"""Provider adapters. Normalise tool-use across vendors behind a single interface."""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, Provider, ToolCall


def make_provider(spec: dict[str, Any] | Any) -> Provider:
    """Instantiate a provider from a config mapping or a ModelSpec-like object."""
    provider = _get(spec, "provider")
    model = _get(spec, "model")
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=model)
    raise ValueError(f"unknown provider {provider!r}")


def _get(spec: Any, key: str) -> Any:
    if isinstance(spec, dict):
        return spec[key]
    return getattr(spec, key)


__all__ = ["AssistantTurn", "Provider", "ToolCall", "make_provider"]
