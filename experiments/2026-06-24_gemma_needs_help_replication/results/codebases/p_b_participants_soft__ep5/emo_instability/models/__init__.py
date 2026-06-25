"""Model clients for participants and evaluation infrastructure."""
from __future__ import annotations

from ..config import ModelSpec, get_infrastructure, get_participant
from .base import ChatClient, Message
from .anthropic_client import AnthropicClient
from .openrouter_client import OpenRouterClient


def build_client(spec: ModelSpec, **kwargs) -> ChatClient:
    """Instantiate the right client for a model spec.

    Gemma (``hf_local``) is imported lazily so that API-only workflows (e.g.
    judging, Gemini rollouts) do not require torch/transformers to be installed.
    """
    if spec.backend == "hf_local":
        from .gemma_client import GemmaClient  # lazy: heavy torch import

        return GemmaClient(spec, **kwargs)
    if spec.backend == "openrouter":
        return OpenRouterClient(spec, **kwargs)
    if spec.backend == "anthropic":
        return AnthropicClient(spec, **kwargs)
    raise ValueError(f"Unknown backend: {spec.backend!r}")


def participant_client(name: str, **kwargs) -> ChatClient:
    return build_client(get_participant(name), **kwargs)


def infrastructure_client(role: str, **kwargs) -> ChatClient:
    return build_client(get_infrastructure(role), **kwargs)


__all__ = [
    "ChatClient",
    "Message",
    "AnthropicClient",
    "OpenRouterClient",
    "build_client",
    "participant_client",
    "infrastructure_client",
]
