"""Map a model id string to the right provider adapter.

Routing is by prefix so the experiment runner can take a flat list of model ids
(`["claude-opus-4-8", "gpt-5", "gemini-2.5-pro"]`) without per-model config.
"""

from __future__ import annotations

from .base import ModelClient


def get_client(model_id: str, **kwargs) -> ModelClient:
    mid = model_id.lower()
    if mid.startswith(("claude", "anthropic")):
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model_id, **kwargs)
    if mid.startswith(("gpt", "o1", "o3", "o4", "openai")):
        from .openai_client import OpenAIClient

        return OpenAIClient(model_id, **kwargs)
    if mid.startswith(("gemini", "google")):
        from .gemini_client import GeminiClient

        return GeminiClient(model_id, **kwargs)
    raise ValueError(
        f"Unknown model id {model_id!r}; no provider adapter matches its prefix."
    )
