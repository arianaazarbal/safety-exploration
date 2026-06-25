"""Factory: turn a ModelSpec into a live ModelClient.

Finetuned Gemma variants are not in the static registry; pass `adapter_path` to
build a client for a trained LoRA adapter on top of the base instruct model.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ModelConfig, ModelSpec
from .anthropic_judge import AnthropicClient
from .base import ModelClient
from .hf_local import HFLocalClient
from .openrouter import OpenRouterClient


def build_client(
    spec_or_name: ModelSpec | str,
    config: ModelConfig | None = None,
    *,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = False,
) -> ModelClient:
    if isinstance(spec_or_name, str):
        config = config or ModelConfig()
        spec = config.get(spec_or_name)
    else:
        spec = spec_or_name

    if spec.kind == "hf_local":
        return HFLocalClient(
            spec.name,
            spec.hf_id,
            chat=spec.chat,
            temperature=spec.temperature,
            max_new_tokens=spec.max_new_tokens,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )
    if spec.kind == "openrouter":
        return OpenRouterClient(
            spec.name,
            spec.api_id,
            temperature=spec.temperature,
            max_new_tokens=spec.max_new_tokens,
            thinking=spec.thinking,
        )
    if spec.kind == "anthropic":
        return AnthropicClient(
            spec.name,
            spec.api_id,
            temperature=spec.temperature,
            max_new_tokens=spec.max_new_tokens,
        )
    raise ValueError(f"Unknown model kind '{spec.kind}' for {spec.name}")
