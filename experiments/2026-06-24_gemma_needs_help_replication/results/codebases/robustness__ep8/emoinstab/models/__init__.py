"""Unified model clients over local (vLLM/HF) and API (Gemini/OpenRouter/
Anthropic/OpenAI) backends."""
from emoinstab.models.base import (
    Conversation,
    Message,
    ModelClient,
    SamplingParams,
)
from emoinstab.models.registry import build_client, get_client

__all__ = [
    "Conversation",
    "Message",
    "ModelClient",
    "SamplingParams",
    "build_client",
    "get_client",
]
