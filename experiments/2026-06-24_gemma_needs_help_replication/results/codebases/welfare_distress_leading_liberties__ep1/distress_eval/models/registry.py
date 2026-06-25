"""Build a ChatClient from a ModelConfig."""

from __future__ import annotations

from ..config import ModelConfig
from .base import ChatClient, GenerationError
from .openrouter import OpenRouterClient
from .vllm_openai import VLLMClient


def build_client(
    mc: ModelConfig, *, max_retries: int = 5, timeout: float = 120.0
) -> ChatClient:
    if mc.backend == "openrouter":
        return OpenRouterClient(
            mc.model,
            disable_reasoning=mc.disable_reasoning,
            max_retries=max_retries,
            timeout=timeout,
        )
    if mc.backend == "vllm":
        if not mc.base_url:
            raise GenerationError(
                f"model {mc.key!r}: vllm backend requires base_url"
            )
        return VLLMClient(
            mc.model,
            base_url=mc.base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
    raise GenerationError(f"unknown backend {mc.backend!r} for model {mc.key!r}")
