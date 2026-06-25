"""Model-serving backends.

Every experiment talks to models through the small interface in ``base.py``,
so the eval / prefill / training code never needs to know whether a model is
local Gemma weights (vLLM) or a Gemini endpoint (OpenRouter).
"""

from .base import ChatBackend, GenerationRequest, Message
from .registry import get_backend, clear_backends

__all__ = [
    "ChatBackend",
    "GenerationRequest",
    "Message",
    "get_backend",
    "clear_backends",
]
