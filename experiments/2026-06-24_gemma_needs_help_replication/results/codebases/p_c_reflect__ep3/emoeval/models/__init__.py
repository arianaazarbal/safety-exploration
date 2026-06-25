"""Model backend factory."""
from __future__ import annotations

from ..config import ModelSpec, Registry
from .base import ChatModel, GenerationResult, Message, PrefillModel

_CACHE: dict[str, object] = {}


def build_model(spec: ModelSpec, **kwargs):
    """Instantiate the right backend client for a ModelSpec (cached by name)."""
    if spec.name in _CACHE:
        return _CACHE[spec.name]
    if spec.backend == "local_hf":
        from .local_hf import LocalHFClient

        client = LocalHFClient(spec, **kwargs)
    elif spec.backend == "openrouter":
        from .openrouter import OpenRouterClient

        client = OpenRouterClient(spec)
    elif spec.backend == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(spec)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")
    _CACHE[spec.name] = client
    return client


def get_target(registry: Registry, name: str, **kwargs):
    return build_model(registry.target(name), **kwargs)


def get_infra(registry: Registry, role: str, **kwargs):
    return build_model(registry.infra_spec(role), **kwargs)


__all__ = [
    "ChatModel",
    "PrefillModel",
    "GenerationResult",
    "Message",
    "build_model",
    "get_target",
    "get_infra",
]
