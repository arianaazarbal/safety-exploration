"""Lazily construct and cache one backend per ModelSpec.

vLLM engines are expensive (they hold GPU memory), so we cache by spec name and
expose `clear_backends()` to free a model before loading the next one - the
Section 2 runner loads models one at a time.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from .base import ChatBackend

if TYPE_CHECKING:  # pragma: no cover
    from ..config import ModelSpec

_CACHE: dict[str, ChatBackend] = {}


def get_backend(spec: "ModelSpec", **kwargs) -> ChatBackend:
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "vllm":
        from .vllm_backend import VLLMBackend
        backend: ChatBackend = VLLMBackend(spec, **kwargs)
    elif spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend
        backend = OpenRouterBackend(spec, **kwargs)
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for {spec.name!r}")

    _CACHE[spec.name] = backend
    return backend


def clear_backends() -> None:
    """Drop cached backends and free GPU memory (vLLM)."""
    for name, backend in list(_CACHE.items()):
        llm = getattr(backend, "llm", None)
        if llm is not None:
            del llm
        del _CACHE[name]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
