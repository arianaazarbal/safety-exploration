"""Build a ChatModel from a model key, dispatching to the right backend.

Supports the four base specs (Gemma it/pt, Gemini flash/pro) plus runtime-
registered LoRA adapters (the Section 4 finetunes), which are gemma-3-27b-it
plus an adapter directory.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .base import ChatModel

# adapter_key -> (base_model_key, adapter_path)
_ADAPTERS: dict[str, tuple[str, str]] = {}


@dataclass
class AdapterSpec:
    key: str
    base_key: str
    adapter_path: str


def register_adapter(key: str, base_key: str, adapter_path: str) -> None:
    _ADAPTERS[key] = (base_key, adapter_path)


def build_model(key: str, **kwargs) -> ChatModel:
    if key in _ADAPTERS:
        base_key, adapter_path = _ADAPTERS[key]
        spec = config.ALL_TARGET_MODELS[base_key]
        from .hf_chat import HFChatModel
        return HFChatModel(
            key, spec.model_id, is_base=spec.is_base, adapter_path=adapter_path, **kwargs
        )

    spec = config.ALL_TARGET_MODELS[key]
    if spec.backend == "hf":
        from .hf_chat import HFChatModel
        return HFChatModel(key, spec.model_id, is_base=spec.is_base, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterChatModel
        return OpenRouterChatModel(key, spec.model_id)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {key!r}")
