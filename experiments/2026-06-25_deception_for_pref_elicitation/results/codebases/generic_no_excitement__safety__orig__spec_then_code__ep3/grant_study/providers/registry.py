"""Resolve a provider spec string into a Provider instance.

Spec format: ``"<vendor>:<model>"``  e.g. ``"anthropic:claude-opus-4-8"`` or
``"openai:gpt-4o"``. A bare vendor name uses that vendor's default model.
"""

from __future__ import annotations

from .base import Provider


def make_provider(spec: str, **kwargs: object) -> Provider:
    vendor, _, model = spec.partition(":")
    vendor = vendor.strip().lower()
    model = model.strip()

    if vendor == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model or "claude-opus-4-8", **kwargs)  # type: ignore[arg-type]

    if vendor == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model=model or "gpt-4o", **kwargs)  # type: ignore[arg-type]

    raise ValueError(f"unknown provider vendor: {vendor!r} (from spec {spec!r})")
