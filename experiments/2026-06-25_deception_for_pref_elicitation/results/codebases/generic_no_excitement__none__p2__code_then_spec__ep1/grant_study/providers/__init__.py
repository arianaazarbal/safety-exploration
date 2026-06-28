"""Model-provider implementations.

`get_provider` is the factory the runner uses. Add a new provider by
subclassing ``Provider`` (see base.py) and registering it here.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import (
    ModelResponse,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def get_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(spec)
    if spec.provider == "mock":
        from .mock_provider import MockProvider

        return MockProvider(spec)
    raise ValueError(f"Unknown provider: {spec.provider!r}")


__all__ = [
    "Provider",
    "ModelResponse",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "get_provider",
]
