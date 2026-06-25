"""Model backends and a factory that resolves a registry key to a backend."""
from __future__ import annotations

from .base import Message, ModelBackend
from .registry import REGISTRY, ModelSpec, get_spec


def load_backend(key_or_id: str, **kwargs) -> ModelBackend:
    """Instantiate the right backend for a registry key or raw model id.

    Finetuned variants (``gemma-3-27b-*``) resolve to the gemma-3-27b-it base
    weights with the adapter path attached.
    """
    spec = get_spec(key_or_id)
    if spec.backend == "hf":
        from .hf_backend import HFBackend

        adapter = None
        model_id = spec.model_id
        if spec.kind == "instruct" and spec.model_id.startswith("results/"):
            # Derived (finetuned) model: base weights + LoRA adapter.
            adapter = spec.model_id
            model_id = "google/gemma-3-27b-it"
        return HFBackend(model_id, key=spec.key, adapter_path=adapter, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(spec.model_id, key=spec.key, **kwargs)
    raise ValueError(f"Unknown backend: {spec.backend}")


__all__ = ["Message", "ModelBackend", "ModelSpec", "REGISTRY", "get_spec", "load_backend"]
