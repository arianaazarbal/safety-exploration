"""Construct a ModelBackend from a config.ModelSpec, lazily importing the
heavy backend (transformers / openai) only when needed.
"""

from __future__ import annotations

from functools import lru_cache

from .base import ModelBackend


def build_backend(spec, **overrides) -> ModelBackend:
    """spec: config.ModelSpec. Extra kwargs are forwarded to the backend ctor."""
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(spec.name, spec.model_id, kind=spec.kind, **overrides)
    if spec.backend == "peft":
        from .hf_backend import HFBackend

        return HFBackend(
            spec.name,
            spec.model_id,
            kind="instruct",
            adapter_path=spec.adapter_path,
            **overrides,
        )
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(spec.name, spec.model_id, **overrides)
    raise ValueError(f"Unknown backend: {spec.backend!r}")
