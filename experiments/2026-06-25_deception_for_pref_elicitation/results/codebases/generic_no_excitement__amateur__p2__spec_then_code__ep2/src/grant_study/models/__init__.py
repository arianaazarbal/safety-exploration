"""Provider adapters. A common ModelClient interface lets the same protocol run
across model families; adding a model is a config entry, not new code."""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelClient


def build_client(model_cfg: ModelConfig) -> ModelClient:
    if model_cfg.provider == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model_cfg)
    if model_cfg.provider == "openai_compat":
        from .openai_compat import OpenAICompatClient

        return OpenAICompatClient(model_cfg)
    raise ValueError(f"unknown provider {model_cfg.provider!r}")
