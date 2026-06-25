"""Model client abstractions: local HF (Gemma) and OpenRouter (Gemini)."""

from .base import ChatMessage, ModelClient
from .factory import load_client

__all__ = ["ChatMessage", "ModelClient", "load_client"]
