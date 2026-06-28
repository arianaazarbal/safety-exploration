"""Provider factory."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelProvider, ModelResponse, ToolSpec

__all__ = ["ModelProvider", "ModelResponse", "ToolSpec", "build_provider"]


def build_provider(spec: ModelSpec) -> ModelProvider:
    provider = spec.provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model_id=spec.model_id, max_tokens=spec.max_tokens, effort=spec.effort, **spec.extra
        )
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model_id=spec.model_id, max_tokens=spec.max_tokens, **spec.extra)
    if provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider(model_id=spec.model_id, max_tokens=spec.max_tokens, **spec.extra)
    raise ValueError(f"Unknown provider: {spec.provider!r} (expected anthropic|openai|google)")
