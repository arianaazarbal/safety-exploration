"""Build a Provider instance from a ModelSpec."""

from __future__ import annotations

from ..config import ModelSpec, require_api_key
from .base import Provider


def build_provider(spec: ModelSpec) -> Provider:
    api_key = require_api_key(spec.provider)

    if spec.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec.model, api_key)

    if spec.provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(spec.model, api_key)

    if spec.provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider(spec.model, api_key)

    raise ValueError(f"Unknown provider: {spec.provider!r}")
