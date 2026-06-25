"""Build a ChatClient from a ModelConfig, with a small cache so heavyweight vLLM
engines are only instantiated once per process."""

from __future__ import annotations

from functools import lru_cache

from ..config import ModelConfig, load_models
from .base import ChatClient

_CACHE: dict[str, ChatClient] = {}


def build_client(model: ModelConfig) -> ChatClient:
    if model.name in _CACHE:
        return _CACHE[model.name]

    if model.backend == "vllm":
        from .vllm_client import VLLMClient

        client: ChatClient = VLLMClient(model)
    elif model.backend == "openrouter":
        from .openrouter_client import OpenRouterClient

        client = OpenRouterClient(model)
    elif model.backend == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(model)
    else:
        raise ValueError(f"Unknown backend: {model.backend}")

    _CACHE[model.name] = client
    return client


@lru_cache(maxsize=1)
def _registry() -> dict[str, ModelConfig]:
    return load_models()


def client_by_name(name: str) -> ChatClient:
    return build_client(_registry()[name])


def model_by_name(name: str) -> ModelConfig:
    return _registry()[name]
