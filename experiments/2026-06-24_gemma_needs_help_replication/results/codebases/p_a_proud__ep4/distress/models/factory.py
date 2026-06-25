"""Factory that turns a model name (or ``ModelSpec``) into a live ``ChatModel``.

Local Gemma models are cached per-process so that, e.g., the judge agreement
pass and the main eval don't reload 27B weights twice. API clients are cheap and
constructed fresh.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelRegistry, ModelSpec
from .base import ChatModel

_HF_CACHE: dict[str, ChatModel] = {}


def build_model(
    name_or_spec: str | ModelSpec,
    registry: ModelRegistry | None = None,
    **backend_kwargs: Any,
) -> ChatModel:
    if isinstance(name_or_spec, ModelSpec):
        spec = name_or_spec
    else:
        registry = registry or ModelRegistry()
        spec = registry.get(name_or_spec)

    backend = spec.backend
    if backend == "hf":
        # Cache key includes adapter path so finetunes don't collide with base.
        key = f"{spec.model_id}::{spec.adapter_path}"
        if key not in _HF_CACHE:
            from .hf_backend import HFBackend

            _HF_CACHE[key] = HFBackend(spec, **backend_kwargs)
        model = _HF_CACHE[key]
        model.name = spec.name  # reflect the requested alias
        return model
    if backend in ("openrouter", "openai"):
        from .openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(spec)
    if backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(spec)
    raise ValueError(f"Unknown backend '{backend}' for model '{spec.name}'.")
