"""Provider-agnostic model adapters (DESIGN.md §5.1)."""
from .base import AssistantTurn, ModelAdapter, ToolCall, ToolSpec
from .registry import get_adapter

__all__ = ["AssistantTurn", "ModelAdapter", "ToolCall", "ToolSpec", "get_adapter"]
