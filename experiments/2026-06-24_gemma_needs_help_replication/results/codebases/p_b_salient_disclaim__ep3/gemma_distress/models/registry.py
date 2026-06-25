"""Resolve a ModelSpec (and optional LoRA adapter) to a concrete ChatModel.

Caches instances so repeated `get_model` calls in one process reuse the loaded
weights / vLLM engine.
"""

from __future__ import annotations

from dataclasses import replace

import config
from config import ModelSpec
from .base import ChatModel

_CACHE: dict[tuple, ChatModel] = {}


def make_finetuned_spec(base: ModelSpec, label: str, backend: str | None = None) -> ModelSpec:
    """Spec for a fine-tuned variant of `base` (e.g. the DPO / SFT Gemma)."""
    return replace(base, name=f"{base.name}-{label}", backend=backend or base.backend)


def get_model(spec: ModelSpec, *, adapter_path: str | None = None,
              backend: str | None = None, **kwargs) -> ChatModel:
    backend = backend or spec.backend
    key = (spec.name, backend, adapter_path)
    if key in _CACHE:
        return _CACHE[key]

    if backend == "hf":
        from .hf import HFChatModel
        model: ChatModel = HFChatModel(spec, adapter_path=adapter_path, **kwargs)
    elif backend == "vllm":
        from .vllm_backend import VLLMChatModel
        model = VLLMChatModel(spec, adapter_path=adapter_path, **kwargs)
    elif backend == "openrouter":
        if adapter_path:
            raise ValueError("Cannot attach a LoRA adapter to an API model.")
        from .openrouter import OpenRouterChatModel
        model = OpenRouterChatModel(spec, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    _CACHE[key] = model
    return model
