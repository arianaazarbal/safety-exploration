"""Map a ModelConfig to a concrete adapter instance."""

from __future__ import annotations

from ..schemas import ModelConfig
from .base import ModelAdapter


def build_adapter(config: ModelConfig) -> ModelAdapter:
    if config.provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(config)
    if config.provider == "openai":
        from .openai_adapter import OpenAIAdapter

        return OpenAIAdapter(config)
    raise ValueError(f"unknown provider: {config.provider!r}")
