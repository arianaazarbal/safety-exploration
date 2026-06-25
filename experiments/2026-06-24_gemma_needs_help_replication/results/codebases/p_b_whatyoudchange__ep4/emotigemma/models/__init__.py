"""Model client backends for target models (Gemma, Gemini)."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel, Message


def build_model(spec: ModelSpec, **kwargs) -> ChatModel:
    """Instantiate the right backend for a model spec.

    Backends are imported lazily so that, e.g., evaluating Gemini does not
    require vLLM/torch to be installed, and vice versa.
    """
    if spec.kind == "gemma_vllm":
        from .gemma_vllm import GemmaVLLMModel
        return GemmaVLLMModel(spec, **kwargs)
    if spec.kind == "gemini":
        from .gemini import GeminiModel
        return GeminiModel(spec, **kwargs)
    raise ValueError(f"Unknown model kind: {spec.kind}")


__all__ = ["ChatModel", "Message", "build_model"]
