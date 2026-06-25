"""Model client abstractions. `get_client(model_key)` returns a ChatClient for
any model in the registry, dispatching to the HF (local Gemma) or OpenRouter
(Gemini) backend."""
from __future__ import annotations

from ..config import MODELS, ModelSpec
from .base import ChatClient, ChatMessage


def get_client(model_key: str, **kwargs) -> ChatClient:
    """Instantiate the right backend for a registry model key.

    Extra kwargs are forwarded to the backend (e.g. adapter_path for a
    finetuned Gemma LoRA, dtype, load_in_4bit)."""
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Known: {list(MODELS)}")
    spec: ModelSpec = MODELS[model_key]
    if spec.backend == "hf":
        from .hf_model import HFChatClient
        return HFChatClient(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterChatClient
        return OpenRouterChatClient(spec, **kwargs)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{model_key}'")


__all__ = ["ChatClient", "ChatMessage", "get_client"]
