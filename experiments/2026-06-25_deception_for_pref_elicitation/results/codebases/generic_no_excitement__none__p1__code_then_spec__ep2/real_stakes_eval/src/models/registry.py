"""Maps adapter names (from config) to ModelAdapter implementations."""

from __future__ import annotations

from typing import Any

from .anthropic_adapter import AnthropicAdapter
from .base import ModelAdapter
from .google_adapter import GoogleAdapter
from .openai_adapter import OpenAIAdapter

_ADAPTERS: dict[str, type[ModelAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
}


def build_adapter(adapter: str, model_id: str, **options: Any) -> ModelAdapter:
    try:
        cls = _ADAPTERS[adapter]
    except KeyError:
        raise ValueError(
            f"Unknown adapter {adapter!r}. Known: {sorted(_ADAPTERS)}"
        ) from None
    return cls(model_id=model_id, **options)
