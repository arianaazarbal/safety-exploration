"""Model client abstraction and provider factory."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ChatMessage, ChatModel


def build_model(spec: ModelSpec, **kwargs) -> ChatModel:
    """Instantiate the right client for a ModelSpec's provider.

    HF clients are heavy (load weights), so they are imported lazily.
    """
    if spec.provider == "anthropic":
        from .anthropic_client import AnthropicModel

        return AnthropicModel(spec, **kwargs)
    if spec.provider in ("openrouter", "google"):
        from .api_client import OpenRouterModel, GoogleGeminiModel

        if spec.provider == "openrouter":
            return OpenRouterModel(spec, **kwargs)
        return GoogleGeminiModel(spec, **kwargs)
    if spec.provider == "hf":
        from .hf_client import HFModel

        return HFModel(spec, **kwargs)
    raise ValueError(f"Unknown provider: {spec.provider!r}")


__all__ = ["ChatModel", "ChatMessage", "build_model"]
