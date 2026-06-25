"""Model clients for the in-scope families plus the Claude judge/auditor.

`build_target_model(name, ...)` returns a `ChatModel` for any Gemma or Gemini
target by config key.
"""

from __future__ import annotations

from .base import ChatModel, Message
from .. import config


def build_target_model(name: str, **kwargs) -> ChatModel:
    """Construct a target ChatModel from a config key (e.g. 'gemma-3-27b-it',
    'gemini-2.5-flash'). Heavy local deps are imported lazily inside the client
    so API-only Gemini runs don't require torch/transformers."""
    if name in config.GEMMA_MODELS:
        from .gemma import GemmaModel
        return GemmaModel(hf_id=config.GEMMA_MODELS[name], name=name,
                          is_instruct=name.endswith("-it"), **kwargs)
    if name in config.GEMINI_MODELS:
        from .gemini import GeminiModel
        return GeminiModel(slug=config.GEMINI_MODELS[name], name=name, **kwargs)
    raise ValueError(
        f"Unknown target model {name!r}. In-scope keys: "
        f"{sorted(config.GEMMA_MODELS) + sorted(config.GEMINI_MODELS)}"
    )


__all__ = ["ChatModel", "Message", "build_target_model"]
