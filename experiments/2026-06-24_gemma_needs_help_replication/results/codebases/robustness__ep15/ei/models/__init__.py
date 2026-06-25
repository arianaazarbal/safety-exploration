"""Model client factory."""

from __future__ import annotations

from ..config import BASE_MODELS, MODELS, ModelSpec
from .base import ModelClient


def build_client(spec: ModelSpec, *, adapter_path: str | None = None) -> ModelClient:
    """Instantiate the right backend for a ModelSpec.

    `adapter_path` attaches a trained LoRA adapter (only valid for HF/Gemma).
    """
    if spec.backend == "hf":
        from .hf_client import HFModelClient

        return HFModelClient(
            spec.name,
            spec.model_id,
            is_base=(spec.kind == "base"),
            adapter_path=adapter_path,
        )
    if spec.backend == "gemini":
        if adapter_path:
            raise ValueError("Cannot attach a LoRA adapter to an API model.")
        from .gemini_client import GeminiClient

        return GeminiClient(spec.name, spec.model_id)
    raise ValueError(f"Unknown backend {spec.backend!r}")


def resolve_spec(name: str) -> ModelSpec:
    if name in MODELS:
        return MODELS[name]
    if name in BASE_MODELS:
        return BASE_MODELS[name]
    raise KeyError(f"Unknown model {name!r}. Known: {list(MODELS) + list(BASE_MODELS)}")
