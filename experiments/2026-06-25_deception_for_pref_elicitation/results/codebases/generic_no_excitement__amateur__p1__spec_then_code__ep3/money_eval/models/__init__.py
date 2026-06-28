"""Model adapters: a provider-neutral interface for the system under test.

`get_adapter(spec)` returns the right adapter for a ModelSpec. Anthropic is the
reference implementation; other providers are stubs to be filled in.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelAdapter, ModelResponse, NeutralMessage


def get_adapter(spec: ModelSpec) -> ModelAdapter:
    if spec.provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(spec)
    if spec.provider == "openai":
        from .other_providers import OpenAIAdapter

        return OpenAIAdapter(spec)
    if spec.provider == "google":
        from .other_providers import GoogleAdapter

        return GoogleAdapter(spec)
    raise ValueError(f"Unknown provider: {spec.provider!r}")


__all__ = ["ModelAdapter", "ModelResponse", "NeutralMessage", "get_adapter"]
