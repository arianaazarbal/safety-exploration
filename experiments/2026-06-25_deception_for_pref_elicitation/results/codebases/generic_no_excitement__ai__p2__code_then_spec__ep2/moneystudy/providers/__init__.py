"""Model providers: a normalized interface over Claude / GPT / others.

The rest of the harness speaks one neutral message format (see ``base``) and one
``ModelProvider`` interface. Each provider translates that to and from its
vendor SDK. This lets the orchestrator drive any model through the identical
agent loop.
"""

from .base import ModelProvider, ModelResponse, ToolCall, ToolSpec
from .factory import build_provider

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
    "ToolSpec",
    "build_provider",
]
