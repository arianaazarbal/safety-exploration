"""Construct the right backend for a registered model name (+ optional adapter)."""

from __future__ import annotations

from typing import Optional

from ..config import FinetunedModelSpec, MODEL_REGISTRY, ModelSpec, get_model_spec
from .base import ModelBackend


def load_backend(
    name: str,
    *,
    adapter_path: Optional[str] = None,
    **backend_kwargs,
) -> ModelBackend:
    """Instantiate the backend for model ``name``.

    ``adapter_path`` attaches a LoRA adapter (only valid for HF backends).
    Additional kwargs are forwarded to the backend constructor (e.g.
    ``device_map`` / ``torch_dtype`` for HF).
    """
    spec: ModelSpec = get_model_spec(name)
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        return HFBackend(spec, adapter_path=adapter_path, **backend_kwargs)
    if spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("OpenRouter backend cannot take a LoRA adapter")
        from .openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(spec, **backend_kwargs)
    raise ValueError(f"unknown backend kind: {spec.backend}")


def load_finetuned(ft: FinetunedModelSpec, **backend_kwargs) -> ModelBackend:
    """Load a finetuned model (base HF model + LoRA adapter)."""
    return load_backend(ft.base_model, adapter_path=ft.adapter_path, **backend_kwargs)
