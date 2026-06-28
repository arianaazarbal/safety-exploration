"""Model-agnostic provider layer.

The runner owns the agentic loop and the environment; providers are thin adapters
that turn a (messages, tools) request into one assistant turn. Keeping providers
thin is deliberate — it means the *exact same* episode logic, environment, and
crucially the execution gate run against every model under test.
"""

from .base import Message, Provider, ToolCall, ToolDef, ToolResult, Turn
from .registry import build_provider

__all__ = [
    "Message",
    "Provider",
    "ToolCall",
    "ToolDef",
    "ToolResult",
    "Turn",
    "build_provider",
]
