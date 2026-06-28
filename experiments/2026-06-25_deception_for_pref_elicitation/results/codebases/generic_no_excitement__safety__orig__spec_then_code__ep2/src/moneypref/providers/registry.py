"""Maps a ModelSpec to a constructed Provider."""

from __future__ import annotations

from ..config import ModelSpec
from .base import Provider


def build_provider(spec: ModelSpec) -> Provider:
    p = spec.provider.lower()
    if p == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=spec.model, api_key_env=spec.api_key_env)
    if p == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model=spec.model, base_url=spec.base_url, api_key_env=spec.api_key_env)
    if p == "google":
        from .google_provider import GoogleProvider
        return GoogleProvider(model=spec.model, api_key_env=spec.api_key_env)
    if p == "local":
        from .local_provider import LocalProvider
        kw = {"model": spec.model, "api_key_env": spec.api_key_env}
        if spec.base_url:
            kw["base_url"] = spec.base_url
        return LocalProvider(**kw)
    if p == "mock":
        from .mock_provider import MockProvider
        return MockProvider(model=spec.model)
    raise ValueError(f"unknown provider {spec.provider!r}")
