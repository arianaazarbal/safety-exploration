"""Model backends: local HuggingFace (Gemma), OpenRouter (Gemini), Anthropic
(Claude judges/auditors)."""

from __future__ import annotations

from config import Backend, ModelSpec
from .base import ChatModel, Message


def load_model(spec: ModelSpec, **kwargs) -> ChatModel:
    """Factory that returns a ready-to-use ChatModel for a ModelSpec."""
    if spec.backend == Backend.HF:
        from .hf_model import HFChatModel

        return HFChatModel(spec, **kwargs)
    if spec.backend == Backend.OPENROUTER:
        from .openrouter_model import OpenRouterChatModel

        return OpenRouterChatModel(spec, **kwargs)
    if spec.backend == Backend.ANTHROPIC:
        from .anthropic_model import AnthropicChatModel

        return AnthropicChatModel(spec, **kwargs)
    raise ValueError(f"Unknown backend: {spec.backend}")


__all__ = ["ChatModel", "Message", "load_model"]
