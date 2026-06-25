"""Model abstractions: a uniform chat/generation interface over local Gemma
(HuggingFace transformers) and Gemini (OpenRouter, OpenAI-compatible)."""
from .base import ChatModel, Message, load_model

__all__ = ["ChatModel", "Message", "load_model"]
