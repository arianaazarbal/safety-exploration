"""Construct a ChatModel from a ModelSpec based on its backend."""
from __future__ import annotations

from ..config import ModelSpec
from .base import ChatModel


def build_model(spec: ModelSpec) -> ChatModel:
    backend = spec.backend.lower()

    if backend == "gemini":
        from .gemini_client import GeminiChatModel
        return GeminiChatModel(spec.key, spec.model, api_key=spec.api_key)

    if backend == "anthropic":
        from .anthropic_client import AnthropicChatModel
        return AnthropicChatModel(spec.key, spec.model, api_key=spec.api_key)

    if backend in ("openai", "openai_compat"):
        from .openai_client import OpenAIChatModel
        return OpenAIChatModel(
            spec.key, spec.model, api_key=spec.api_key, base_url=spec.base_url
        )

    if backend == "hf":
        from .hf_client import HFChatModel
        return HFChatModel(spec.key, spec.model)

    raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.key!r}")
