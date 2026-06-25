"""Model client implementations and a factory keyed on the config registry."""

from __future__ import annotations

from functools import lru_cache

from ..config import MODELS, ModelSpec
from .base import ChatModel, Message


@lru_cache(maxsize=None)
def get_model(key: str, **overrides) -> ChatModel:
    """Return a ChatModel for a registry key, constructing it lazily.

    Heavy local models are only imported/loaded when first requested, so the
    package can be imported (and the API-only experiments run) on a machine
    without GPUs or transformers installed.
    """
    spec: ModelSpec = MODELS[key]
    if spec.backend == "hf":
        from .hf_local import HFChatModel
        return HFChatModel(spec, **overrides)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterChatModel
        return OpenRouterChatModel(spec, **overrides)
    if spec.backend == "anthropic":
        from .anthropic_client import AnthropicChatModel
        return AnthropicChatModel(spec, **overrides)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {key!r}")


__all__ = ["ChatModel", "Message", "get_model"]
