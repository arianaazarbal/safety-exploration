"""Resolve a config.ModelSpec (or registry name) into a live ChatModel.

Local/LoRA backends are cached per process: vLLM holds the weights on GPU, so we
never instantiate the same checkpoint twice. (In practice you evaluate one local
model per process / GPU allocation; Gemini is stateless API access.)
"""
from __future__ import annotations

import config
from .base import ChatModel

_CACHE: dict[str, ChatModel] = {}


def load_model(spec_or_name) -> ChatModel:
    spec = (config.REGISTRY[spec_or_name]
            if isinstance(spec_or_name, str) else spec_or_name)
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "openrouter":
        from .openrouter_client import OpenRouterModel
        model: ChatModel = OpenRouterModel(spec.model_id, spec.name)
    elif spec.backend in ("local", "lora"):
        from .local_backend import LocalModel
        model = LocalModel(
            model_id=spec.model_id,
            name=spec.name,
            is_base=spec.is_base,
            adapter_path=spec.adapter_path,
        )
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")

    _CACHE[spec.name] = model
    return model
