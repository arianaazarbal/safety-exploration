"""Resolve a registry name to a concrete, instantiated `ModelClient`.

Backends are imported lazily so that, e.g., running the Section 2 evaluation of
the Gemini models does not require torch/transformers to be installed.
"""
from __future__ import annotations

import config
from .base import ModelClient


def load_model(name: str, **backend_kwargs) -> ModelClient:
    if name not in config.REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Known: {sorted(config.REGISTRY)}"
        )
    spec = config.REGISTRY[name]

    if spec.backend == "huggingface":
        from .huggingface_backend import HuggingFaceClient
        return HuggingFaceClient(spec, **backend_kwargs)
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterClient
        return OpenRouterClient(spec, **backend_kwargs)

    raise ValueError(f"unsupported backend '{spec.backend}' for model '{name}'")
