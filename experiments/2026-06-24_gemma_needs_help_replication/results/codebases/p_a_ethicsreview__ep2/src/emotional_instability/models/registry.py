"""Resolve a model name -> ChatModel instance, with caching.

Local backends (hf/vllm) are heavyweight (GPU memory); we cache one instance per
name per process so repeated `build_model` calls don't reload weights.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec, load_model_registry
from .base import ChatModel

_REGISTRY: dict[str, ModelSpec] | None = None


def _registry() -> dict[str, ModelSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_model_registry()
    return _REGISTRY


def get_spec(name: str) -> ModelSpec:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(reg)}")
    return reg[name]


@lru_cache(maxsize=None)
def build_model(name: str) -> ChatModel:
    spec = get_spec(name)
    if spec.backend == "hf":
        from .hf_model import HFModel

        return HFModel(spec)
    if spec.backend == "vllm":
        from .vllm_model import VLLMModel

        return VLLMModel(spec)
    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterModel

        return OpenRouterModel(spec)
    if spec.backend == "anthropic":
        from .anthropic_model import AnthropicModel

        return AnthropicModel(spec)
    if spec.backend == "openai":
        from .openai_model import OpenAIModel

        return OpenAIModel(spec)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {name!r}")
