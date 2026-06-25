"""Model client implementations and a factory keyed on the model registry."""

from __future__ import annotations

from ..config import Backend, ModelSpec, FinetunedSpec, GenerationConfig
from .base import ModelClient, ChatMessage, GenerationResult
from .gemma_local import GemmaLocalClient
from .gemini_api import GeminiOpenRouterClient
from .anthropic_judge import AnthropicClient


_CACHE: dict[str, ModelClient] = {}


def get_client(spec: ModelSpec, gen: GenerationConfig | None = None) -> ModelClient:
    """Return (and cache) a client for ``spec``.

    Local models are expensive to load, so clients are memoised by key.
    """
    if spec.key in _CACHE:
        return _CACHE[spec.key]

    if spec.backend is Backend.HF_LOCAL:
        adapter = getattr(spec, "adapter_path", None)
        client: ModelClient = GemmaLocalClient(spec, gen=gen, adapter_path=adapter)
    elif spec.backend is Backend.OPENROUTER:
        client = GeminiOpenRouterClient(spec, gen=gen)
    elif spec.backend is Backend.ANTHROPIC:
        client = AnthropicClient(spec.model_id)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown backend: {spec.backend}")

    _CACHE[spec.key] = client
    return client


__all__ = [
    "ModelClient",
    "ChatMessage",
    "GenerationResult",
    "GemmaLocalClient",
    "GeminiOpenRouterClient",
    "AnthropicClient",
    "get_client",
]
