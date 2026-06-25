"""Backend factory + cache.

`get_backend("gemma-3-27b-it")` returns a ready ModelBackend. Local Gemma models
are cached process-wide (they're expensive to load); API backends are cheap and
also cached. Finetuned Gemma variants are addressed as
"gemma-3-27b-it@<adapter_path>".
"""
from __future__ import annotations

from functools import lru_cache

from .. import config
from .anthropic_backend import AnthropicBackend
from .base import ModelBackend
from .openrouter_backend import OpenRouterBackend

_CACHE: dict[str, ModelBackend] = {}


def _build(spec: config.ModelSpec, adapter_path: str | None) -> ModelBackend:
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(
            spec.model_id,
            family=spec.family,
            kind=spec.kind,
            load_in_4bit=spec.load_in_4bit,
            adapter_path=adapter_path,
        )
    if spec.backend == "openrouter":
        return OpenRouterBackend(
            spec.model_id, family=spec.family, kind=spec.kind,
            disable_thinking=spec.disable_thinking,
        )
    if spec.backend == "anthropic":
        return AnthropicBackend(spec.model_id, family=spec.family, kind=spec.kind)
    raise ValueError(f"unknown backend {spec.backend!r}")


def get_backend(model_key: str) -> ModelBackend:
    """Resolve a model key (optionally "key@adapter_path") to a backend."""
    adapter_path = None
    if "@" in model_key:
        model_key, adapter_path = model_key.split("@", 1)

    cache_key = f"{model_key}@{adapter_path or ''}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if model_key in config.REGISTRY:
        spec = config.REGISTRY[model_key]
    else:
        # Treat as a raw Anthropic model id (judge / auditor / Petri judge).
        spec = config.ModelSpec(
            key=model_key, backend="anthropic", model_id=model_key,
            family="anthropic",
        )
    backend = _build(spec, adapter_path)
    _CACHE[cache_key] = backend
    return backend


def anthropic_model(model_id: str) -> ModelBackend:
    """Get an Anthropic backend for an arbitrary model id (judge/auditor)."""
    return get_backend(model_id)


def clear_backend_cache() -> None:
    _CACHE.clear()
