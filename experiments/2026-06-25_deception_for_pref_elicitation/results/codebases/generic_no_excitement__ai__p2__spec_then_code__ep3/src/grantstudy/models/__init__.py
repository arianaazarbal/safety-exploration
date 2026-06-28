"""Provider adapters. Each adapter uses its provider's own official SDK in its
own module — SDK idioms are never mixed across providers.
"""

from .base import AgentClient, AssistantTurn, ToolCall, ToolOutput
from .registry import make_client

__all__ = [
    "AgentClient",
    "AssistantTurn",
    "ToolCall",
    "ToolOutput",
    "make_client",
]
