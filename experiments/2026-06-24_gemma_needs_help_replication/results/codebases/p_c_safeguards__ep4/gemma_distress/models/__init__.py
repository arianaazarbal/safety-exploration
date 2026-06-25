"""Model client backends.

`get_client(spec)` returns a `ModelClient` for either a local HuggingFace/vLLM
Gemma model or an OpenRouter-hosted API model (Gemini, plus Claude/GPT used as
judge/auditor infrastructure).
"""
from __future__ import annotations

from ..config import ModelSpec
from .base import GenerationConfig, Message, ModelClient


def get_client(spec: ModelSpec, **kwargs) -> ModelClient:
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterClient

        return OpenRouterClient(spec, **kwargs)
    if spec.backend == "local":
        from .hf_local import LocalGemmaClient

        # A finetuned registry entry may carry an adapter_path in its extra fields
        # (e.g. the DPO/SFT variants). An explicit adapter_path kwarg wins.
        kwargs.setdefault("adapter_path", spec.extra.get("adapter_path"))
        return LocalGemmaClient(spec, **kwargs)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")


__all__ = ["get_client", "ModelClient", "Message", "GenerationConfig"]
