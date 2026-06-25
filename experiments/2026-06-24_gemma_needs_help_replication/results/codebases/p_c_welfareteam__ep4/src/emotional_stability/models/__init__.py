"""Model backends: local Gemma (HF), Gemini (OpenRouter), Claude (Anthropic)."""

from emotional_stability.models.base import ChatModel, GenerationConfig
from emotional_stability.models.registry import get_chat_model

__all__ = ["ChatModel", "GenerationConfig", "get_chat_model"]
