"""Model client factory (Gemma local + Gemini OpenRouter)."""

from __future__ import annotations

import config
from .base import ChatModel, Message
from .gemini import GeminiOpenRouterModel
from .gemma import GemmaLocalModel

_CACHE: dict[str, ChatModel] = {}


def get_model(key: str, **kwargs) -> ChatModel:
    """Return (and cache) a ChatModel for a registry key or a finetuned-variant key.

    Finetuned variants (``gemma-3-27b-it-dpo`` etc.) load the base 27B-it model and
    attach a LoRA adapter from ``config.FINETUNE_DIR/<key>``.
    """
    if key in _CACHE:
        return _CACHE[key]

    if key in config.TARGET_MODELS:
        spec = config.TARGET_MODELS[key]
        if spec.backend == "hf_local":
            model = GemmaLocalModel(spec, **kwargs)
        elif spec.backend == "openrouter":
            model = GeminiOpenRouterModel(spec, **kwargs)
        else:
            raise ValueError(f"unknown backend {spec.backend}")
    elif key.startswith("gemma-3-27b-it-"):
        # Finetuned variant: base 27B-it + LoRA adapter directory.
        base_spec = config.TARGET_MODELS["gemma-3-27b-it"]
        adapter_dir = config.FINETUNE_DIR / key
        model = GemmaLocalModel(base_spec, adapter_path=str(adapter_dir),
                                display_key=key, **kwargs)
    else:
        raise KeyError(f"unknown model key: {key}")

    _CACHE[key] = model
    return model


__all__ = ["ChatModel", "Message", "get_model"]
