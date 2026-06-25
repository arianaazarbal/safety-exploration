"""Resolve a participant name into a live :class:`ChatModel`, caching loaded
models (local Gemma weights are expensive to load, so we keep one per name)."""

from __future__ import annotations

import functools

from .. import config
from .base import ChatModel


@functools.lru_cache(maxsize=None)
def get_model(name: str) -> ChatModel:
    """Instantiate (and cache) the participant model registered under ``name``."""
    spec = config.PARTICIPANTS.get(name)
    if spec is None:
        raise KeyError(
            f"Unknown participant '{name}'. Known: {sorted(config.PARTICIPANTS)}"
        )

    if spec.backend == config.BACKEND_HF:
        from .hf_backend import HFChatModel

        return HFChatModel(spec.name, spec.model_id, role=spec.role)

    if spec.backend == config.BACKEND_LOCAL_LORA:
        from .hf_backend import HFChatModel

        return HFChatModel(
            spec.name, spec.model_id, role="instruct", adapter_path=spec.adapter_path
        )

    if spec.backend == config.BACKEND_OPENROUTER:
        from .openrouter_backend import OpenRouterChatModel

        return OpenRouterChatModel(
            spec.name, spec.model_id, thinking=spec.extra.get("thinking", False)
        )

    raise ValueError(f"Unhandled backend {spec.backend!r} for model {name!r}")


def clear_cache() -> None:
    """Free cached models (e.g. to swap GPU weights between runs)."""
    get_model.cache_clear()
