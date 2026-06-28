"""Provider-agnostic model adapters. The only provider-specific code lives here."""

from .base import (
    AssistantTurn,
    ModelAdapter,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .registry import build_adapter

__all__ = [
    "AssistantTurn",
    "ModelAdapter",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "build_adapter",
]
