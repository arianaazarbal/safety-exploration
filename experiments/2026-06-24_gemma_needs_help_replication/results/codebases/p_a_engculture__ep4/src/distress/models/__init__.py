"""Model provider abstraction.

A ``ModelProvider`` turns a chat conversation into an assistant message. The
factory ``load_provider`` maps a :class:`distress.config.ModelSpec` to a concrete
backend. Heavy backends (transformers, vLLM, anthropic, openai) are imported
lazily so importing this package never forces a torch/CUDA load.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import GenConfig, Message, ModelProvider

__all__ = ["GenConfig", "Message", "ModelProvider", "load_provider"]


def load_provider(spec: ModelSpec, **kwargs) -> ModelProvider:
    """Instantiate the concrete provider for ``spec``."""
    if spec.provider == "hf":
        from .local_hf import HFProvider

        return HFProvider(spec, **kwargs)
    if spec.provider == "vllm":
        from .vllm_provider import VLLMProvider

        return VLLMProvider(spec, **kwargs)
    if spec.provider == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(spec, **kwargs)
    if spec.provider == "anthropic":
        from .anthropic_client import AnthropicProvider

        return AnthropicProvider(spec, **kwargs)
    if spec.provider == "openai":
        from .openai_client import OpenAIProvider

        return OpenAIProvider(spec, **kwargs)
    raise ValueError(f"Unknown provider {spec.provider!r} for {spec.key}")
