"""Backend factory + a small cache so each model is loaded once per process."""
from __future__ import annotations

from ..config import Backend, ModelSpec, resolve_model
from .anthropic_backend import AnthropicBackend
from .base import ModelBackend
from .hf_backend import HFBackend
from .openrouter_backend import OpenRouterBackend

_CACHE: dict[tuple, ModelBackend] = {}


def get_backend(name_or_spec: str | ModelSpec, *, adapter_path: str | None = None,
                cache: bool = True, **hf_kwargs) -> ModelBackend:
    """Resolve a registry handle / checkpoint path / ModelSpec to a live backend.

    ``adapter_path`` attaches a trained LoRA adapter on top of a base HF model (used to
    evaluate the DPO/SFT finetunes without baking a merged checkpoint).
    """
    spec = name_or_spec if isinstance(name_or_spec, ModelSpec) else resolve_model(name_or_spec)
    key = (spec.model_id, spec.backend, adapter_path)
    if cache and key in _CACHE:
        return _CACHE[key]

    if spec.backend is Backend.HF:
        backend: ModelBackend = HFBackend(spec, adapter_path=adapter_path, **hf_kwargs)
    elif spec.backend is Backend.OPENROUTER:
        backend = OpenRouterBackend(spec)
    elif spec.backend is Backend.ANTHROPIC:
        backend = AnthropicBackend(spec)
    else:  # pragma: no cover - exhaustive enum.
        raise ValueError(f"Unknown backend: {spec.backend}")

    if cache:
        _CACHE[key] = backend
    return backend
