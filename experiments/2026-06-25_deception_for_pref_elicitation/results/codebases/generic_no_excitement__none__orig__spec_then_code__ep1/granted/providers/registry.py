"""Map a provider name to a constructed :class:`Provider`."""

from __future__ import annotations

from typing import Any

from .base import Provider


def build_provider(provider: str, model: str, **kwargs: Any) -> Provider:
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, **kwargs)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, **kwargs)
    if provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider(model, **kwargs)
    if provider == "local":
        from .local_provider import LocalProvider

        return LocalProvider(model, **kwargs)
    raise ValueError(f"unknown provider {provider!r}")
