"""Model client abstraction.

`build_client(name, settings)` returns a `ChatModel` for any model in
config/models.yaml. Local Gemma models are served with vLLM (preferred) or
transformers; Gemini models via OpenRouter.
"""
from __future__ import annotations

from .base import ChatModel, Message
from .factory import build_client

__all__ = ["ChatModel", "Message", "build_client"]
