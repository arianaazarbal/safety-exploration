"""Unified chat-model interface and backend factory."""
from __future__ import annotations

import config
from .base import ChatModel, Message, GenerationParams


def load_model(spec: "config.ModelSpec", adapter_path: str | None = None, **kw) -> ChatModel:
    """Construct the right backend for ``spec``.

    Imports are lazy so that, e.g., running the API-only judge does not require
    torch/transformers to be installed, and vice versa.
    """
    if spec.backend == "hf":
        from .hf_model import HFModel
        return HFModel(spec, adapter_path=adapter_path, **kw)
    if spec.backend == "openrouter":
        from .api_model import OpenRouterModel
        return OpenRouterModel(spec, **kw)
    if spec.backend == "anthropic":
        from .api_model import AnthropicModel
        return AnthropicModel(spec, **kw)
    raise ValueError(f"Unknown backend: {spec.backend!r}")


__all__ = ["ChatModel", "Message", "GenerationParams", "load_model"]
