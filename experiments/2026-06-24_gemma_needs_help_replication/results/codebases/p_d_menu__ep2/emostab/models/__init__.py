"""Unified model-client interface over the backends used in the replication.

  get_client(spec) -> ModelClient

Subjects:
  - Gemma -> HFClient (local transformers; supports chat, completion, prefill,
             continuation, and residual-stream capture for Appendix I)
  - Gemini -> OpenRouterClient (OpenAI-compatible)
Infrastructure:
  - Claude judge/auditor/utility -> AnthropicClient
  - GPT-5-mini validation judge -> OpenAIClient
"""
from __future__ import annotations

from ..config import Backend, ModelSpec
from .base import ChatMessage, GenerationResult, ModelClient

_CACHE: dict[str, ModelClient] = {}


def get_client(spec: ModelSpec, **kwargs) -> ModelClient:
    """Return a (cached) client for the given model spec."""
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == Backend.HF:
        from .hf_client import HFClient
        client: ModelClient = HFClient(spec, **kwargs)
    elif spec.backend == Backend.OPENROUTER:
        from .openrouter_client import OpenRouterClient
        client = OpenRouterClient(spec, **kwargs)
    elif spec.backend == Backend.ANTHROPIC:
        from .anthropic_client import AnthropicClient
        client = AnthropicClient(spec, **kwargs)
    elif spec.backend == Backend.OPENAI:
        from .openai_client import OpenAIClient
        client = OpenAIClient(spec, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {spec.backend}")

    _CACHE[spec.name] = client
    return client


__all__ = ["ChatMessage", "GenerationResult", "ModelClient", "get_client"]
