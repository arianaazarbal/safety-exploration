"""Resolve a model key (from config) to a concrete ChatModel instance."""
from __future__ import annotations

import functools

import config
from .base import ChatModel


def is_local_model(key: str) -> bool:
    return key in config.GEMMA_MODELS


@functools.lru_cache(maxsize=None)
def _load_cached(key: str, adapter_path: str | None, load_in_4bit: bool) -> ChatModel:
    if key in config.GEMMA_MODELS:
        from .gemma import GemmaModel
        return GemmaModel(key, config.GEMMA_MODELS[key],
                          adapter_path=adapter_path, load_in_4bit=load_in_4bit)
    if key in config.GEMINI_MODELS:
        from .gemini import GeminiModel
        return GeminiModel(key, config.GEMINI_MODELS[key])
    raise KeyError(f"Unknown model key '{key}'. Known: "
                   f"{sorted(config.GEMMA_MODELS) + sorted(config.GEMINI_MODELS)}")


def load_model(key: str, *, adapter_path: str | None = None,
               load_in_4bit: bool = False) -> ChatModel:
    """Instantiate (and cache) a model by config key.

    ``adapter_path`` attaches a LoRA adapter (local Gemma only); used to load the
    DPO/SFT-finetuned models from Section 4.
    """
    return _load_cached(key, adapter_path, load_in_4bit)
