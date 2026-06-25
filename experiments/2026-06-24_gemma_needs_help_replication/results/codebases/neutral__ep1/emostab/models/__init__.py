"""Chat-model clients and a factory.

A `Message` is a dict ``{"role": "user"|"assistant"|"system", "content": str}``.
A `Conversation` is a list of `Message`. All target models implement `ChatModel`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import ModelSpec, get_model
from .base import ChatModel, Conversation, Message

if TYPE_CHECKING:  # avoid importing heavy deps at module load
    pass

__all__ = ["ChatModel", "Conversation", "Message", "load_model"]


def load_model(key_or_spec: str | ModelSpec, **kwargs) -> ChatModel:
    """Instantiate the right backend for a model key or spec.

    Heavy backends (torch/vllm) are imported lazily so that lightweight code
    paths (prompt construction, analysis) do not pay for them.
    """
    spec = get_model(key_or_spec) if isinstance(key_or_spec, str) else key_or_spec

    if spec.backend == "vllm":
        from .local import VLLMModel
        return VLLMModel(spec, **kwargs)
    if spec.backend == "hf":
        from .local import HFModel
        return HFModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
