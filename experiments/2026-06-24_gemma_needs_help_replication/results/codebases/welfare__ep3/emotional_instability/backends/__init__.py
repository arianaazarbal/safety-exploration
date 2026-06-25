"""Model backends: a uniform chat interface over OpenRouter (API) and local
HuggingFace inference, plus the Anthropic-based judge/auditor client."""
from .base import ChatBackend, ChatMessage, get_backend

__all__ = ["ChatBackend", "ChatMessage", "get_backend"]
