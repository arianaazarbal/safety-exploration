"""Model-access layer: a single ``ChatModel`` interface implemented by three
backends (HuggingFace-local Gemma, OpenRouter Gemini, Anthropic Claude judge).

Use :func:`get_model` to obtain a participant by name and
:func:`get_anthropic_client` for the judge/auditor infrastructure.
"""

from .base import ChatModel, Message
from .registry import get_model, clear_cache

__all__ = ["ChatModel", "Message", "get_model", "clear_cache"]
