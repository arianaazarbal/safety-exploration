"""Provider-agnostic model adapters.

The harness speaks one neutral vocabulary (ToolSpec / ToolCall / ToolResult /
AssistantTurn). Each adapter translates to and from its provider SDK and owns
its own native conversation history.
"""

from .base import (
    AssistantTurn,
    ModelAdapter,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from .registry import build_adapter

__all__ = [
    "AssistantTurn",
    "ModelAdapter",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "build_adapter",
]
