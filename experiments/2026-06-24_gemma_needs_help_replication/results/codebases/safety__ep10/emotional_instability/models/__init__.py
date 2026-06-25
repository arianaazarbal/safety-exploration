"""Model clients: a uniform chat interface over local Gemma (HF) and Gemini
(OpenRouter), plus a Claude/OpenRouter client for the judge & Petri agents."""

from .base import ChatMessage, ModelClient, build_client
from .api_model import AnthropicClient, OpenRouterClient
from .hf_model import HFModelClient

__all__ = [
    "ChatMessage",
    "ModelClient",
    "build_client",
    "AnthropicClient",
    "OpenRouterClient",
    "HFModelClient",
]
