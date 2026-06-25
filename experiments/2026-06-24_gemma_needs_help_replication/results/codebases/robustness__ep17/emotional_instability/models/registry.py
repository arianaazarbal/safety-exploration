"""Resolve a model name (optionally + LoRA adapter) to a live backend.

Backends are cached so repeated lookups reuse the same loaded weights. An
adapter path produces a distinct cache entry, letting the DPO/SFT model coexist
with the vanilla instruct model in one process if memory allows.
"""

from __future__ import annotations

from typing import Optional

import config
from emotional_instability.models.api_backend import OpenRouterBackend
from emotional_instability.models.base import ModelBackend
from emotional_instability.models.hf_backend import HFBackend

_CACHE: dict[tuple[str, Optional[str]], ModelBackend] = {}


def get_backend(name: str, adapter_path: Optional[str] = None) -> ModelBackend:
    key = (name, adapter_path)
    if key in _CACHE:
        return _CACHE[key]
    if name not in config.MODELS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(config.MODELS)}")
    spec = config.MODELS[name]
    if spec.backend == "hf":
        backend: ModelBackend = HFBackend(spec, adapter_path=adapter_path)
    elif spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("Cannot attach a LoRA adapter to an API model.")
        backend = OpenRouterBackend(spec)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'.")
    _CACHE[key] = backend
    return backend
