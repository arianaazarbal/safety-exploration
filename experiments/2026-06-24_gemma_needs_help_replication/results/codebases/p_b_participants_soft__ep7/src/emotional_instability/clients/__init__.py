"""Model client abstraction.

A single interface (`ModelClient.chat`) hides whether a model is served via
OpenRouter (Gemini participants + Claude/GPT instruments + cloud Gemma) or run
locally via HuggingFace transformers (Gemma base/instruct continuations and
LoRA-finetuned adapters).
"""
from .base import ChatMessage, GenerationResult, ModelClient
from .registry import get_client

__all__ = ["ChatMessage", "GenerationResult", "ModelClient", "get_client"]
