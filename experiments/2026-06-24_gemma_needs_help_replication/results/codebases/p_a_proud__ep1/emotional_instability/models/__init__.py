"""Unified chat-model clients.

`build_client(spec)` returns a :class:`ChatModel` for any registered
:class:`~emotional_instability.config.ModelSpec`, dispatching on its backend:

* ``hf``         -> :class:`HFChatModel`        (Gemma, local GPU)
* ``openrouter`` -> :class:`OpenAICompatModel`  (Gemini)
* ``openai``     -> :class:`OpenAICompatModel`  (GPT-5-mini judge)
* ``anthropic``  -> :class:`AnthropicModel`     (Claude judge / Petri)

All clients share the :class:`ChatModel` surface so the eval, prefill, training
and Petri code is backend-agnostic.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel, Message, GenResult

_HF_CACHE: dict[str, ChatModel] = {}


def build_client(spec: ModelSpec, *, adapter_path: str | None = None, **kwargs) -> ChatModel:
    """Construct (or, for local HF models, reuse) a chat client for ``spec``.

    Local HF models are cached by (model_id, adapter_path) because loading a 27B
    checkpoint is expensive; API clients are cheap and created fresh.
    """
    if spec.backend == "hf":
        cache_key = f"{spec.model_id}::{adapter_path or ''}"
        if cache_key not in _HF_CACHE:
            from .hf_model import HFChatModel
            _HF_CACHE[cache_key] = HFChatModel(spec, adapter_path=adapter_path, **kwargs)
        return _HF_CACHE[cache_key]
    if spec.backend in ("openrouter", "openai"):
        from .api_model import OpenAICompatModel
        return OpenAICompatModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .api_model import AnthropicModel
        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.key!r}")


__all__ = ["ChatModel", "Message", "GenResult", "build_client"]
