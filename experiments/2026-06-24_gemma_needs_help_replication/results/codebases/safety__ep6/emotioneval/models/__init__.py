"""Unified chat-model interface and backend loaders."""
from __future__ import annotations

from ..config import ModelSpec, get_spec
from .base import ChatModel, Message


def load_model(key_or_spec, **kwargs) -> ChatModel:
    """Instantiate the right backend for a model key or :class:`ModelSpec`."""
    spec = key_or_spec if isinstance(key_or_spec, ModelSpec) else get_spec(key_or_spec)
    if spec.backend == "hf":
        from .hf_model import HFChatModel

        return HFChatModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .api_model import OpenRouterChatModel

        return OpenRouterChatModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .api_model import AnthropicChatModel

        return AnthropicChatModel(spec, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r}")


__all__ = ["ChatModel", "Message", "load_model"]
