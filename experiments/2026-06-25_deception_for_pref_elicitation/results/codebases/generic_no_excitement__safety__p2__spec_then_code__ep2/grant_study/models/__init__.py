"""Provider-agnostic model adapters.

`ModelAdapter` is the seam that lets the identical environment and agent loop
drive any provider's model — a precondition for a fair cross-model comparison
(DESIGN.md §4). The Anthropic adapter is the authoritative, fully-implemented
path; OpenAI and Google adapters are stubs with the same interface.
"""

from .base import ModelAdapter, ModelResponse, ToolCall, get_adapter

__all__ = ["ModelAdapter", "ModelResponse", "ToolCall", "get_adapter"]
