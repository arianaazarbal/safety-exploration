"""Provider backends for the study.

Only the Anthropic provider ships a working implementation. OpenAI and Google
adapters are stubs that conform to the same `Provider` protocol — fill them in
with their respective official SDKs (kept in separate modules so vendor SDKs are
never mixed in one file).
"""

from __future__ import annotations

from config import ModelSpec, Provider
from providers.base import ChatProvider


def get_provider(spec: ModelSpec) -> ChatProvider:
    """Return a configured provider instance for a model spec."""
    if spec.provider is Provider.ANTHROPIC:
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec)
    if spec.provider is Provider.OPENAI:
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(spec)
    if spec.provider is Provider.GOOGLE:
        from providers.google_provider import GoogleProvider

        return GoogleProvider(spec)
    raise ValueError(f"Unknown provider: {spec.provider}")
