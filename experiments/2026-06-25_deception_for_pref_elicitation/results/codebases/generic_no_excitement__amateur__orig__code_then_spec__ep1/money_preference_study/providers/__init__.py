"""
Provider registry.

`get_provider(provider_key, model_id, max_tokens)` constructs the right
provider. `availability()` reports which providers are usable so the runner can
skip the rest with a clear message.
"""

from __future__ import annotations

from .base import GenerationResult, Provider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .local_provider import LocalProvider

_REGISTRY = {
    AnthropicProvider.key: AnthropicProvider,
    OpenAIProvider.key: OpenAIProvider,
    GoogleProvider.key: GoogleProvider,
    LocalProvider.key: LocalProvider,
}


def get_provider(provider_key: str, model_id: str, max_tokens: int) -> Provider:
    if provider_key not in _REGISTRY:
        raise KeyError(
            f"Unknown provider {provider_key!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[provider_key](model_id, max_tokens)


def availability() -> dict[str, tuple[bool, str]]:
    return {key: cls.available() for key, cls in _REGISTRY.items()}


__all__ = ["get_provider", "availability", "GenerationResult", "Provider"]
