"""Construct a ModelClient from a ModelSpec."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelClient


def load_client(spec: ModelSpec, *, adapter_dir: str | None = None, **hf_kwargs) -> ModelClient:
    if spec.backend == "hf":
        from .hf_model import HFChatModel

        return HFChatModel(
            spec.key,
            spec.model_id,
            is_base=spec.is_base,
            adapter_dir=adapter_dir,
            **hf_kwargs,
        )
    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterChatModel

        return OpenRouterChatModel(spec.key, spec.model_id)
    raise ValueError(f"Unknown backend: {spec.backend}")
