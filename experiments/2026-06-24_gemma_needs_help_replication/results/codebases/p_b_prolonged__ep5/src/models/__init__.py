"""Unified chat/generation interface across local Gemma, OpenRouter Gemini, and
Anthropic Claude judges. ``get_model(spec)`` returns a ``ChatModel``."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel, Message


def get_model(spec: ModelSpec, **kwargs) -> ChatModel:
    """Factory dispatching on ``spec.backend``. Lazily imports the backend so
    that, e.g., running a Gemini-only experiment does not require torch."""
    if spec.backend == "hf":
        from .hf_local import HFLocalModel
        return HFLocalModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_client import AnthropicModel
        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"unknown backend: {spec.backend!r}")


__all__ = ["get_model", "ChatModel", "Message"]
