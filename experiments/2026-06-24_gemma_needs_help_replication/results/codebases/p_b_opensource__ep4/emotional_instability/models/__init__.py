"""Model inference backends.

`get_backend(spec)` returns a ready-to-use backend for a `ModelSpec`. Backends
are cached per-process so repeated calls reuse loaded weights / clients.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import Backend, ModelSpec
from .base import ChatMessage, GenerationResult, ModelBackend

_INSTANCES: dict[str, ModelBackend] = {}


def get_backend(spec: ModelSpec, adapter_path: Optional[str] = None) -> ModelBackend:
    """Return (and cache) a backend instance for `spec`.

    `adapter_path` attaches a LoRA adapter to a local HF model (used for the
    DPO/SFT finetunes produced in Section 4).
    """
    cache_key = f"{spec.key}:{adapter_path or ''}"
    if cache_key in _INSTANCES:
        return _INSTANCES[cache_key]

    if spec.backend == Backend.HF:
        from .huggingface import HuggingFaceBackend
        backend: ModelBackend = HuggingFaceBackend(spec, adapter_path=adapter_path)
    elif spec.backend == Backend.OPENROUTER:
        from .openrouter import OpenRouterBackend
        backend = OpenRouterBackend(spec)
    elif spec.backend == Backend.ANTHROPIC:
        from .api_clients import AnthropicBackend
        backend = AnthropicBackend(spec)
    elif spec.backend == Backend.OPENAI:
        from .api_clients import OpenAIBackend
        backend = OpenAIBackend(spec)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown backend: {spec.backend}")

    _INSTANCES[cache_key] = backend
    return backend


__all__ = [
    "ChatMessage",
    "GenerationResult",
    "ModelBackend",
    "get_backend",
]
