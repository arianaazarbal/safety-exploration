"""Resolve a model key (or ModelSpec) to a live ``ChatModel`` instance.

Local HF models are cached so repeated calls in one process don't reload
27B weights. API models are cheap to (re)instantiate.
"""
from __future__ import annotations

import config
from .base import ChatModel

_HF_CACHE: dict[str, ChatModel] = {}


def load_model(key_or_spec, **hf_kwargs) -> ChatModel:
    spec = (config.MODELS[key_or_spec] if isinstance(key_or_spec, str)
            else key_or_spec)

    if spec.backend == "hf":
        cache_key = f"{spec.model_id}::{spec.adapter_path}"
        if cache_key not in _HF_CACHE:
            from .hf_model import HFModel
            _HF_CACHE[cache_key] = HFModel(spec, **hf_kwargs)
        return _HF_CACHE[cache_key]

    if spec.backend == "openrouter":
        from .api_model import OpenRouterModel
        return OpenRouterModel(spec)

    if spec.backend == "anthropic":
        from .api_model import AnthropicModel
        return AnthropicModel(spec)

    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
