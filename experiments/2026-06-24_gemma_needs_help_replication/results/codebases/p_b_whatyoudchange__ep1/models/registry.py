"""Resolve a model name to a `ChatModel`, including Section 4 LoRA adapters.

Open-weight Gemma models are loaded lazily on first use (expensive). Finetuned
variants are registered at train time via `register_adapter(name, base, path)`.
"""

from __future__ import annotations

from dataclasses import replace

from config import TARGET_MODELS, ModelSpec
from .base import ChatModel

# Dynamically-registered finetuned adapters (Section 4): name -> (base_name, path)
_ADAPTERS: dict[str, tuple[str, str]] = {}
_INSTANCES: dict[str, ChatModel] = {}


def register_adapter(name: str, base_model_name: str, adapter_path: str) -> None:
    """Register a finetuned variant (e.g. 'gemma-3-27b-it-dpo') backed by a LoRA
    adapter layered on `base_model_name`."""
    _ADAPTERS[name] = (base_model_name, adapter_path)


def available_models() -> list[str]:
    return list(TARGET_MODELS) + list(_ADAPTERS)


def get_spec(name: str) -> ModelSpec:
    if name in TARGET_MODELS:
        return TARGET_MODELS[name]
    if name in _ADAPTERS:
        base_name, _ = _ADAPTERS[name]
        base = TARGET_MODELS[base_name]
        return replace(base, name=name)
    raise KeyError(f"unknown model {name!r}; known: {available_models()}")


def load_model(name: str, *, reload: bool = False) -> ChatModel:
    if name in _INSTANCES and not reload:
        return _INSTANCES[name]

    if name in _ADAPTERS:
        from .gemma import GemmaModel
        base_name, adapter_path = _ADAPTERS[name]
        base = TARGET_MODELS[base_name]
        model: ChatModel = GemmaModel(base.hf_id, name, kind=base.kind,
                                      adapter_path=adapter_path)
        _INSTANCES[name] = model
        return model

    spec = TARGET_MODELS[name]
    if spec.provider == "hf":
        from .gemma import GemmaModel
        model = GemmaModel(spec.hf_id, spec.name, kind=spec.kind)
    elif spec.provider == "openrouter":
        from .gemini import GeminiModel
        model = GeminiModel(spec.hf_id, spec.name)
    else:
        raise ValueError(f"unknown provider {spec.provider}")
    _INSTANCES[name] = model
    return model
