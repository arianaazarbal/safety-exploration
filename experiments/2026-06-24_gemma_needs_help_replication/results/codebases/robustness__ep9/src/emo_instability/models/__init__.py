"""Model-client abstraction.

A single interface (:class:`ModelClient`) covers the three access patterns we
need:

* local Gemma instruct models for batched elicitation sweeps (vLLM)  -> ``VLLMClient``
* local Gemma instruct/base models for prefill/continuation + training-time
  generation (HuggingFace transformers)                              -> ``HFClient``
* API models (Gemini via OpenRouter, Anthropic judge/auditor)        -> ``OpenAICompatClient`` / ``AnthropicClient``

Use :func:`build_client` to construct the right backend from a ``ModelSpec``.
"""
from __future__ import annotations

from .base import ChatMessage, GenerationError, ModelClient
from .factory import build_client

__all__ = ["ChatMessage", "ModelClient", "GenerationError", "build_client"]
