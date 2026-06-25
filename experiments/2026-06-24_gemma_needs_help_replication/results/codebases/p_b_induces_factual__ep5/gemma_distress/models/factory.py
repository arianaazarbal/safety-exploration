"""Resolve a friendly model key to a loaded ModelClient."""

from __future__ import annotations

from .. import config
from .base import ModelClient


def load_model(key: str, *, adapter_path: str | None = None, **kwargs) -> ModelClient:
    """Load a target model by its config key.

    Recognised keys: anything in GEMMA_MODELS, GEMMA_BASE_MODEL, GEMINI_MODELS,
    or an explicit HF repo id. `adapter_path` attaches a LoRA adapter (Gemma
    only) so the trained DPO/SFT mitigation can be evaluated through the exact
    same code path as the vanilla model.
    """
    if key in config.GEMINI_MODELS:
        from .gemini import GeminiClient

        return GeminiClient(config.GEMINI_MODELS[key], name=key)

    if key in config.GEMMA_MODELS or key in config.GEMMA_BASE_MODEL:
        from .gemma import GemmaClient

        model_id = {**config.GEMMA_MODELS, **config.GEMMA_BASE_MODEL}[key]
        return GemmaClient(model_id, name=key, adapter_path=adapter_path, **kwargs)

    # Fall back to treating `key` as a raw HF repo id (e.g. a custom checkpoint).
    from .gemma import GemmaClient

    return GemmaClient(key, name=key, adapter_path=adapter_path, **kwargs)
