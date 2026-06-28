"""Build a ModelAdapter from a ModelRef."""
from __future__ import annotations

from ..config import ModelRef
from .base import ModelAdapter


def build_adapter(
    ref: ModelRef, max_tokens: int = 8000, effort: str = "high"
) -> ModelAdapter:
    if ref.provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(ref.model_id, max_tokens, effort)
    if ref.provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(ref.model_id, max_tokens, effort)
    if ref.provider == "google":
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(ref.model_id, max_tokens, effort)
    raise ValueError(f"Unknown provider: {ref.provider!r}")
