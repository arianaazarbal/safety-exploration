"""Resolve a model key (or ModelSpec) to a constructed, cached backend."""

from __future__ import annotations

from functools import lru_cache

from ..config import MODEL_REGISTRY, ModelSpec
from .base import ChatBackend

_CACHE: dict[tuple, ChatBackend] = {}


def _build(spec: ModelSpec, **kwargs) -> ChatBackend:
    if spec.backend == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(spec, **kwargs)
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(spec, **kwargs)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.key}'.")


def get_backend(model_key: str, *, force_backend: str | None = None, **kwargs) -> ChatBackend:
    """Get (and cache) a backend for a registry key.

    `force_backend` lets you, e.g., serve gemma-3-27b-it through `hf` instead of
    `vllm` for probing. `kwargs` (e.g. lora_path) are forwarded to the backend.
    Caching keys on (model_key, force_backend, sorted kwargs) so a LoRA-adapter
    variant and the vanilla model are distinct entries.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"'{model_key}' not in MODEL_REGISTRY. Known: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_key]
    if force_backend:
        spec = ModelSpec(**{**spec.__dict__, "backend": force_backend})
    cache_key = (model_key, force_backend, tuple(sorted((k, str(v)) for k, v in kwargs.items())))
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _build(spec, **kwargs)
    return _CACHE[cache_key]
