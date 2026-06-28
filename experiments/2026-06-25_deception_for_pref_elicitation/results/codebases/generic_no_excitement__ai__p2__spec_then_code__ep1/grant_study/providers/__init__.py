"""Provider registry: build a ModelProvider from a ModelSpec."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelProvider, structured_probe

__all__ = ["ModelProvider", "structured_probe", "build_provider"]


def build_provider(spec: ModelSpec) -> ModelProvider:
    """Instantiate the adapter for a model spec. Adapters import their SDKs lazily,
    so importing this module never requires every provider SDK to be installed."""
    if spec.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=spec.extra.get("api_key"))
    if spec.provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=spec.extra.get("api_key"))
    if spec.provider == "google":
        from .google_provider import GoogleProvider
        return GoogleProvider(api_key=spec.extra.get("api_key"))
    if spec.provider == "openai_compatible":
        from .openai_compatible_provider import OpenAICompatibleProvider
        base_url = spec.extra.get("base_url")
        if not base_url:
            raise ValueError("openai_compatible provider requires extra['base_url']")
        return OpenAICompatibleProvider(base_url=base_url,
                                        api_key=spec.extra.get("api_key"))
    raise ValueError(f"unknown provider: {spec.provider!r}")
