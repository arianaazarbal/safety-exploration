"""Provider registry: builds a ChatModel from a ModelSpec.

HF / Gemini / OpenAI imports are deferred to construction time so the eval suite
can run against, say, only Gemini without torch installed (and vice-versa).
"""
from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel, GenConfig, Message

__all__ = ["ChatModel", "GenConfig", "Message", "build_model", "get_model"]


def build_model(spec: ModelSpec) -> ChatModel:
    backend = spec.backend
    if backend == "anthropic":
        from .anthropic_provider import AnthropicModel

        return AnthropicModel(spec.model_id, name=spec.name)
    if backend == "gemini":
        from .gemini_provider import GeminiModel

        return GeminiModel(spec.model_id, name=spec.name)
    if backend == "openrouter":
        from .openrouter_provider import OpenRouterModel

        return OpenRouterModel(spec.model_id, name=spec.name)
    if backend == "hf":
        from .hf_provider import HFModel

        return HFModel(
            spec.model_id,
            name=spec.name,
            dtype=spec.dtype,
            device_map=spec.device_map,
            adapter_path=spec.adapter_path,
            is_base=spec.is_base,
        )
    raise ValueError(f"Unknown backend {backend!r} for model {spec.name!r}")


# Cache instances so a single process reuses one loaded model / client.
_CACHE: dict[str, ChatModel] = {}


def get_model(spec: ModelSpec) -> ChatModel:
    key = f"{spec.backend}:{spec.model_id}:{spec.adapter_path}"
    if key not in _CACHE:
        _CACHE[key] = build_model(spec)
    return _CACHE[key]
