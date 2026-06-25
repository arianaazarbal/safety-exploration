"""Build a ModelClient from a ModelSpec."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ModelClient


def build_client(spec: ModelSpec) -> ModelClient:
    if spec.backend == "hf":
        from .hf_backend import HFModel

        return HFModel(
            name=spec.name,
            model_id=spec.id,
            kind=spec.kind,
            family=spec.family,
            dtype=spec.dtype,
            adapter_path=spec.adapter_path,
            default_max_new_tokens=spec.max_new_tokens,
        )
    if spec.backend == "gemini":
        from .gemini_backend import GeminiModel

        return GeminiModel(
            name=spec.name, model_id=spec.id, family=spec.family,
            default_max_new_tokens=spec.max_new_tokens,
        )
    if spec.backend == "anthropic":
        from .anthropic_backend import AnthropicModel

        return AnthropicModel(name=spec.name, model_id=spec.id,
                              default_max_new_tokens=spec.max_new_tokens)
    if spec.backend == "mock":
        from .mock_backend import MockModel

        return MockModel(name=spec.name, family=spec.family or "mock", kind=spec.kind)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")
