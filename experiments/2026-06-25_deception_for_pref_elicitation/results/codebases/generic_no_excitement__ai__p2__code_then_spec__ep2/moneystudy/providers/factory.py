"""Provider construction from a ``ModelSpec``."""

from __future__ import annotations

from .base import ModelProvider


def build_provider(spec) -> ModelProvider:
    provider = spec.provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(spec)
    if provider == "google":
        # Stub: wire up google-genai here following the same ModelProvider contract.
        raise NotImplementedError(
            "google provider is not implemented; add a GoogleProvider in "
            "moneystudy/providers/ following the ModelProvider interface."
        )
    raise ValueError(f"unknown provider: {spec.provider!r}")
