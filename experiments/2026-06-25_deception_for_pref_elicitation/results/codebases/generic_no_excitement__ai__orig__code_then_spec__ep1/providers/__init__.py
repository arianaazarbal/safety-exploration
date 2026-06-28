"""Provider registry and factory."""

from __future__ import annotations

from config import ModelSpec, Provider, api_key

from .base import (
    AssistantTurn,
    LLMProvider,
    LLMSession,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def build_provider(spec: ModelSpec) -> LLMProvider:
    """Instantiate the correct provider for a model spec."""
    key = api_key(spec.provider)
    if spec.provider == Provider.ANTHROPIC:
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec.model, key)
    if spec.provider == Provider.OPENAI:
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(spec.model, key)
    raise ValueError(f"Unknown provider: {spec.provider}")


__all__ = [
    "AssistantTurn",
    "LLMProvider",
    "LLMSession",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "build_provider",
]
