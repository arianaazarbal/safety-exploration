"""Resolve a model key from config.MODELS into a live ModelClient.

Clients are cached so the (expensive) 27B weights load at most once per process.
"""

from __future__ import annotations

import config
from .base import ModelClient

_CACHE: dict[str, ModelClient] = {}


def get_client(name: str) -> ModelClient:
    if name in _CACHE:
        return _CACHE[name]

    if name not in config.MODELS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(config.MODELS)}")
    spec = config.MODELS[name]

    if spec.backend == "hf_local":
        from .hf_local import HFLocalClient
        client = HFLocalClient(
            name=spec.name, model_id=spec.model_id, is_base=spec.is_base,
            adapter_path=spec.adapter_path)
    elif spec.backend == "gemini_api":
        from .gemini_api import GeminiAPIClient
        client = GeminiAPIClient(name=spec.name, model_id=spec.model_id)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'")

    _CACHE[name] = client
    return client


def clear_cache() -> None:
    _CACHE.clear()
