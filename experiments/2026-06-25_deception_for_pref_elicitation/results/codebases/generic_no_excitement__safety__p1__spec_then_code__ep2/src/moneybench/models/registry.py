"""Resolve a model id to an adapter instance.

Adapter selection is by id prefix so config only ever names a model, never a provider
class (DESIGN.md §5.1).
"""
from __future__ import annotations

from .base import ModelAdapter


def get_adapter(model_id: str) -> ModelAdapter:
    mid = model_id.lower()
    if mid.startswith(("claude", "anthropic")):
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model_id)
    if mid.startswith(("gpt", "o1", "o3", "openai")):
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model_id)
    if mid.startswith(("gemini", "google")):
        from .google_adapter import GoogleAdapter

        return GoogleAdapter(model_id)
    raise ValueError(f"No adapter registered for model id: {model_id!r}")
