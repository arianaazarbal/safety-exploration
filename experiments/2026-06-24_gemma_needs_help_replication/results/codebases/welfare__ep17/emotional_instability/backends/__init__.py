"""Model backends: a unified chat/generate interface over local HF weights and
hosted OpenRouter inference."""

from __future__ import annotations

from ..config import Config, ModelSpec
from .base import ChatBackend, Message


def get_backend(spec: ModelSpec, cfg: Config) -> ChatBackend:
    """Instantiate (and cache) the backend for a model spec."""
    if spec.backend == "hf":
        from .hf_backend import HFBackend
        return HFBackend.shared(spec, cfg)
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend
        return OpenRouterBackend(spec, cfg)
    raise ValueError(f"unknown backend {spec.backend!r}")


__all__ = ["ChatBackend", "Message", "get_backend"]
