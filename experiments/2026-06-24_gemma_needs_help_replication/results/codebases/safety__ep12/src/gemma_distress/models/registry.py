"""Factory mapping a ModelSpec to a concrete backend instance (cached)."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ModelBackend

_CACHE: dict[str, ModelBackend] = {}


def get_backend(spec: ModelSpec, **kwargs) -> ModelBackend:
    cache_key = f"{spec.name}|{spec.adapter or ''}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if spec.backend == "vllm":
        from .vllm_backend import VLLMBackend

        backend: ModelBackend = VLLMBackend(spec, **kwargs)
    elif spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend

        backend = OpenRouterBackend(spec, **kwargs)
    elif spec.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        backend = AnthropicBackend(spec, **kwargs)
    else:
        raise ValueError(f"unknown backend '{spec.backend}' for model '{spec.name}'")

    _CACHE[cache_key] = backend
    return backend
