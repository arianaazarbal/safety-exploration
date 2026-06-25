"""Model-access backends (OpenAI-compatible: OpenRouter + local vLLM)."""
from .base import ChatBackend, GenResult, Message
from .registry import close_all, get_backend

__all__ = ["ChatBackend", "GenResult", "Message", "get_backend", "close_all"]
