"""Unified chat-model interface and backends.

The rest of the codebase talks to models through :class:`ChatModel`.  Backends:

- :class:`gemma_distress.models.huggingface.HFChatModel` -- local transformers,
  supports response prefilling and residual-stream / logit capture (required
  for Section 3 and Appendix I).
- :class:`gemma_distress.models.vllm_backend.VLLMChatModel` -- fast batched
  sampling of local Gemma; supports prefill via chat-template continuation.
- :class:`gemma_distress.models.openrouter.OpenRouterChatModel` -- Gemini (and
  the GPT-5-mini cross-check judge) via the OpenAI-compatible OpenRouter API.

Use :func:`build_model` to construct a backend from a :class:`ModelConfig`.
"""
from __future__ import annotations

from ..config import ModelConfig
from .base import ChatModel, Message


def build_model(cfg: ModelConfig) -> ChatModel:
    """Instantiate the backend named by ``cfg.backend``.

    Heavy backends are imported lazily so that, e.g., running the API-only
    Gemini evaluations does not require torch/vLLM to be installed.
    """
    backend = cfg.backend.lower()
    if backend == "hf":
        from .huggingface import HFChatModel

        return HFChatModel(cfg)
    if backend == "vllm":
        from .vllm_backend import VLLMChatModel

        return VLLMChatModel(cfg)
    if backend in ("openrouter", "openai"):
        from .openrouter import OpenRouterChatModel

        return OpenRouterChatModel(cfg)
    raise ValueError(f"Unknown model backend {cfg.backend!r}")


__all__ = ["ChatModel", "Message", "build_model"]
