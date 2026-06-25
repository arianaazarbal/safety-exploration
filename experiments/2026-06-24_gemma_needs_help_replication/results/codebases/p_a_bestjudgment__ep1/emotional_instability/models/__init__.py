"""Model backends for generation.

`get_backend(spec)` returns a ready-to-use backend for a ModelSpec. Backends
are cached per process so vLLM only loads weights once.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import Backend, ModelSpec
from .base import ModelBackend


@lru_cache(maxsize=None)
def get_backend(spec: ModelSpec) -> ModelBackend:
    if spec.backend == Backend.VLLM:
        from .vllm_backend import VLLMBackend
        return VLLMBackend(spec)
    if spec.backend == Backend.OPENROUTER:
        from .openrouter_backend import OpenRouterBackend
        return OpenRouterBackend(spec)
    raise ValueError(f"unknown backend {spec.backend}")
