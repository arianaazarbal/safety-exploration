"""Model client backends.

A single `ModelClient` interface abstracts over local Gemma (HF / vLLM) and
API models (Gemini via OpenRouter, Claude/GPT judges). The eval and training
code never branches on backend; it only calls `chat()` / `complete()` /
`generate_n()`.
"""
from __future__ import annotations

from typing import Any

from .base import ChatMessage, GenerationConfig, ModelClient


def build_client(spec: dict[str, Any]) -> ModelClient:
    """Construct a client from a model/judge config dict (must include `kind`)."""
    kind = spec["kind"]
    if kind == "hf":
        from .hf_local import HFClient

        return HFClient(spec)
    if kind == "vllm":
        # vLLM is preferred for the thousands of samples the eval needs, but we
        # gracefully fall back to transformers if it is not installed so the
        # pipeline still runs (just slower).
        try:
            from .vllm_local import VLLMClient

            return VLLMClient(spec)
        except Exception:  # pragma: no cover - environment dependent
            from .hf_local import HFClient

            return HFClient(spec)
    if kind == "openai":
        from .api import OpenAICompatClient

        return OpenAICompatClient(spec)
    if kind == "anthropic":
        from .api import AnthropicClient

        return AnthropicClient(spec)
    raise ValueError(f"Unknown model kind: {kind!r}")


__all__ = [
    "ChatMessage",
    "GenerationConfig",
    "ModelClient",
    "build_client",
]
