"""Backend factory + a small in-process cache.

Local backends (hf/vllm) are expensive to instantiate (they load weights), so we
cache one instance per model name. API backends are cheap but we cache anyway for
consistency.
"""
from __future__ import annotations

from ..config import ModelSpec

_CACHE: dict[str, object] = {}


def build_client(spec: ModelSpec, *, cache: bool = True, **kwargs):
    if cache and spec.name in _CACHE:
        return _CACHE[spec.name]

    backend = spec.backend
    if backend == "hf":
        from .huggingface import HuggingFaceClient

        client = HuggingFaceClient(spec, **kwargs)
    elif backend == "vllm":
        from .vllm_client import VLLMClient

        client = VLLMClient(spec, **kwargs)
    elif backend == "gemini":
        from .gemini import GeminiClient

        client = GeminiClient(spec, **kwargs)
    elif backend == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(spec, **kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{spec.name}'")

    if cache:
        _CACHE[spec.name] = client
    return client


def clear_cache():
    _CACHE.clear()
