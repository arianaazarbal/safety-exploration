"""Build a Provider from a subject/judge spec."""

from __future__ import annotations

from ..models import ToolSpec
from .base import Provider


def build_provider(
    provider: str,
    model: str,
    system_prompt: str,
    tools: list[ToolSpec],
    *,
    effort: str = "high",
    max_tokens: int = 16000,
) -> Provider:
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model, system_prompt, tools, effort=effort, max_tokens=max_tokens
        )
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, system_prompt, tools, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider {provider!r}")
