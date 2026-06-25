"""Build a :class:`ChatClient` from a :class:`ModelSpec`, with caching."""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec
from .anthropic_client import AnthropicClient
from .base import ChatClient
from .openrouter_client import OpenRouterClient


@lru_cache(maxsize=16)
def _cached_client(backend: str, model_id: str, name: str, adapter_path: str | None,
                   is_base: bool) -> ChatClient:
    if backend == "hf":
        from .hf_client import HFClient  # imported lazily (heavy torch deps)

        return HFClient(model_id, name, adapter_path=adapter_path, is_base=is_base)
    if backend == "openrouter":
        return OpenRouterClient(model_id, name)
    if backend == "anthropic":
        return AnthropicClient(model_id, name)
    raise ValueError(f"unknown backend {backend!r}")


def get_client(spec: ModelSpec, *, adapter_path: str | None = None) -> ChatClient:
    """Return a (cached) client for `spec`.

    `adapter_path` attaches a LoRA adapter (HF only) for evaluating finetunes.
    """
    return _cached_client(
        spec.backend, spec.model_id, spec.name, adapter_path, spec.role == "base"
    )
