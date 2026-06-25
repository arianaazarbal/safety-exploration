"""Build a subject backend from a ModelSpec."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ModelBackend


def get_backend(spec: ModelSpec, **overrides) -> ModelBackend:
    """Instantiate the backend for a subject model spec.

    `overrides` lets callers pass e.g. load_in_4bit=True for the HF backend.
    """
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(
            name=spec.name,
            hf_id=spec.hf_id,
            is_chat=spec.is_chat,
            adapter_path=spec.adapter_path,
            **overrides,
        )
    if spec.backend == "openrouter":
        from .gemini_backend import OpenRouterBackend

        return OpenRouterBackend(
            name=spec.name,
            api_id=spec.api_id,
            thinking=bool(spec.get("thinking", False)),
            is_chat=spec.is_chat,
        )
    raise ValueError(f"Unknown subject backend: {spec.backend}")
