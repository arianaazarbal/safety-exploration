"""Maps a provider name to its adapter. Add new providers here."""

from __future__ import annotations

from .base import AgentClient


def make_client(provider: str, model_id: str) -> AgentClient:
    if provider == "anthropic":
        from .anthropic_client import AnthropicAgentClient

        return AnthropicAgentClient(model_id)
    if provider == "openai":
        from .openai_client import OpenAIAgentClient

        return OpenAIAgentClient(model_id)
    raise ValueError(f"unknown provider: {provider!r}")
