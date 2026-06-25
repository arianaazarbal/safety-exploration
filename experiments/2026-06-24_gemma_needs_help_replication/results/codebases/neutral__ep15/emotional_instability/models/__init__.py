"""Model client abstractions and a factory keyed on :class:`config.ModelSpec`."""
from __future__ import annotations

from config import ModelSpec
from .base import ChatClient, Message

_CACHE: dict[str, ChatClient] = {}


def get_client(spec: ModelSpec) -> ChatClient:
    """Return a (process-cached) client for ``spec``.

    Local backends are cached because loading a 27B model is expensive; we never
    want two copies resident at once.
    """
    if spec.key in _CACHE:
        return _CACHE[spec.key]

    if spec.backend == "openrouter":
        from .openrouter import OpenRouterClient
        client: ChatClient = OpenRouterClient(spec)
    elif spec.backend == "vllm":
        from .vllm_local import VLLMClient
        client = VLLMClient(spec)
    elif spec.backend == "hf":
        from .hf_local import HFClient
        client = HFClient(spec)
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"unknown backend: {spec.backend!r}")

    _CACHE[spec.key] = client
    return client


__all__ = ["ChatClient", "Message", "get_client"]
