"""Model clients.

The rest of the codebase only depends on the :class:`ChatModel` interface, so
the elicitation/judge/training code is agnostic to whether a target runs
locally (Gemma via transformers) or behind an API (Gemini via google.genai).
"""

from .base import ChatModel, Message, prefill_supported
from .registry import load_model

__all__ = ["ChatModel", "Message", "prefill_supported", "load_model"]
