"""Provider registry and factory."""

from __future__ import annotations

from ..config import ProviderConfig, resolve_api_key
from .base import Message, Provider, ProviderResponse, ToolCall, ToolSchema

__all__ = [
    "Message",
    "Provider",
    "ProviderResponse",
    "ToolCall",
    "ToolSchema",
    "build_provider",
]


def build_provider(cfg: ProviderConfig) -> Provider:
    """Instantiate the right Provider subclass from a ProviderConfig."""

    api_key = resolve_api_key(cfg)

    if cfg.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg.name, cfg.model, api_key=api_key, **cfg.extra)

    if cfg.provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(cfg.name, cfg.model, api_key=api_key, base_url=cfg.base_url, **cfg.extra)

    if cfg.provider == "openai_compatible":
        from .openai_provider import OpenAICompatibleProvider

        if not cfg.base_url:
            raise ValueError(f"provider '{cfg.name}' is openai_compatible but has no base_url")
        return OpenAICompatibleProvider(cfg.name, cfg.model, base_url=cfg.base_url, api_key=api_key, **cfg.extra)

    if cfg.provider == "google":
        from .google_provider import GoogleProvider

        return GoogleProvider(cfg.name, cfg.model, api_key=api_key, **cfg.extra)

    raise ValueError(f"unknown provider type: {cfg.provider!r}")
