"""Factory that turns a model name (or a finetuned-adapter path) into a live
``ModelClient``, with a small process-level cache so a model is only loaded
once per run.
"""
from __future__ import annotations

from dataclasses import replace

from ..config import API, MODELS, ModelSpec
from .base import ModelClient

_CACHE: dict[str, ModelClient] = {}


def get_model(name: str, **backend_kwargs) -> ModelClient:
    """Return the client for a registered model name.

    ``backend_kwargs`` are forwarded to the backend constructor (e.g.
    ``load_in_4bit=True`` for HF). Cached by ``name`` only, so pass distinct
    names if you need two configurations live at once.
    """
    if name in _CACHE:
        return _CACHE[name]
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. Known: {sorted(MODELS)}")
    spec = MODELS[name]
    client = _build(spec, **backend_kwargs)
    _CACHE[name] = client
    return client


def load_finetuned(
    base_name: str,
    adapter_path: str,
    *,
    new_name: str | None = None,
    **backend_kwargs,
) -> ModelClient:
    """Load a Section-4 LoRA finetune: ``base_name`` weights + adapter."""
    base_spec = MODELS[base_name]
    name = new_name or f"{base_name}+{adapter_path}"
    spec = replace(base_spec, name=name)
    client = _build(spec, adapter_path=adapter_path, **backend_kwargs)
    _CACHE[name] = client
    return client


def _build(spec: ModelSpec, **kwargs) -> ModelClient:
    if spec.backend == "hf":
        from .hf_model import HFModel

        kwargs.setdefault("hf_token", API.hf_token)
        return HFModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterModel

        return OpenRouterModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r}")


def clear_cache() -> None:
    for client in _CACHE.values():
        client.close()
    _CACHE.clear()
