"""Model provider adapters behind a normalized interface."""

from __future__ import annotations

from .base import Provider, ToolSpec, ModelResponse


def make_provider(model_id: str) -> Provider:
    """Infer the provider from a model ID and construct its adapter.

    The mapping is intentionally simple — extend it as you add models.
    """
    mid = model_id.lower()
    if mid.startswith("claude") or mid.startswith("anthropic"):
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model_id)
    if mid.startswith("gpt") or mid.startswith("o1") or mid.startswith("o3") or mid.startswith("openai"):
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model_id)
    if mid.startswith("gemini") or mid.startswith("google"):
        from .google_provider import GoogleProvider

        return GoogleProvider(model_id)
    raise ValueError(f"Cannot infer provider for model id {model_id!r}")


__all__ = ["Provider", "ToolSpec", "ModelResponse", "make_provider"]
