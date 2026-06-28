"""Provider registry.

`build_provider` maps a provider key to a concrete `ModelProvider`. Add new backends by
implementing `ModelProvider` and registering them here.
"""

from __future__ import annotations

from .base import Message, ModelProvider, ProviderResponse

_REGISTRY = {}


def _register(key: str, factory) -> None:
    _REGISTRY[key] = factory


def build_provider(
    provider: str, model_id: str, *, max_tokens: int = 16000, effort: str = "high"
) -> ModelProvider:
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[provider](model_id, max_tokens=max_tokens, effort=effort)


# Lazy factories so importing this package doesn't require every provider SDK.
def _anthropic_factory(model_id, **kw):
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider(model_id, **kw)


def _openai_factory(model_id, **kw):
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(model_id, **kw)


def _google_factory(model_id, **kw):
    from .google_provider import GoogleProvider

    return GoogleProvider(model_id, **kw)


def _local_factory(model_id, **kw):
    from .local_provider import LocalProvider

    return LocalProvider(model_id, **kw)


_register("anthropic", _anthropic_factory)
_register("openai", _openai_factory)
_register("google", _google_factory)
_register("local", _local_factory)

__all__ = ["Message", "ModelProvider", "ProviderResponse", "build_provider"]
