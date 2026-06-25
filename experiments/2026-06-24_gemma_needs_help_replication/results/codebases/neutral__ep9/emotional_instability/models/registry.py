"""Backend factory with a small instance cache.

Local HF models are expensive to load, so backends are memoised by model name.
The cache can be cleared between heavy phases (e.g. before swapping the 27B
model for a finetuned variant) to free GPU memory.
"""
from __future__ import annotations

import config
from .base import ModelBackend

_CACHE: dict[str, ModelBackend] = {}


def get_backend(name: str) -> ModelBackend:
    if name in _CACHE:
        return _CACHE[name]
    spec = config.MODEL_REGISTRY[name]
    if spec.backend == "hf":
        from .hf_backend import HFBackend
        backend: ModelBackend = HFBackend(spec)
    elif spec.backend == "openrouter":
        from .api_backend import OpenRouterBackend
        backend = OpenRouterBackend(spec)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'")
    _CACHE[name] = backend
    return backend


def clear_backend_cache() -> None:
    """Drop cached backends and free GPU memory if torch is loaded."""
    _CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
