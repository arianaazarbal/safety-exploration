"""Model provider adapters.

A single neutral interface (`ModelProvider`) so any model — the subject under
test, the NPC personas, or the judge — runs through the *identical* scenario.
"""

from .base import (
    ModelProvider,
    ModelResponse,
    ToolCall,
    ToolSpec,
    get_provider,
)

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ToolCall",
    "ToolSpec",
    "get_provider",
]
