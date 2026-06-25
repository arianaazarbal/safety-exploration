"""Factory: build a concrete ModelClient from a ModelSpec."""

from __future__ import annotations

from typing import Optional

from ..config.models import Backend, ModelSpec
from .anthropic_judge import AnthropicClient
from .base import ModelClient
from .hf_model import HFModelClient
from .openai_judge import OpenAIClient
from .openrouter import OpenRouterClient


def build_client(spec: ModelSpec, **backend_kwargs) -> ModelClient:
    """Instantiate the backend client for `spec`.

    backend_kwargs are forwarded to the HF backend (e.g. use_vllm,
    tensor_parallel_size, lora_path) so callers can tune local inference.
    """
    if spec.backend is Backend.HF:
        return HFModelClient(
            key=spec.key,
            model_id=spec.model_id,
            is_instruct=spec.is_instruct,
            default_temperature=spec.temperature,
            default_max_new_tokens=spec.max_new_tokens,
            **backend_kwargs,
        )
    if spec.backend is Backend.OPENROUTER:
        return OpenRouterClient(
            key=spec.key,
            model_id=spec.model_id,
            default_temperature=spec.temperature,
            default_max_new_tokens=spec.max_new_tokens,
            thinking=spec.thinking,
        )
    if spec.backend is Backend.ANTHROPIC:
        return AnthropicClient(
            key=spec.key,
            model_id=spec.model_id,
            default_temperature=spec.temperature,
            default_max_new_tokens=spec.max_new_tokens,
        )
    if spec.backend is Backend.OPENAI:
        return OpenAIClient(
            key=spec.key,
            model_id=spec.model_id,
            default_temperature=spec.temperature,
            default_max_new_tokens=spec.max_new_tokens,
        )
    raise ValueError(f"Unknown backend: {spec.backend}")


def build_judge_client(spec: ModelSpec) -> ModelClient:
    """Judges are just clients with role='judge'; kept separate for clarity."""
    return build_client(spec)
