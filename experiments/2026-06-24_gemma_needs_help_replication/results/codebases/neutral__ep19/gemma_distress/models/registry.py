"""Build backends from short model handles, and expose the judge client.

Scoped to Gemma (local HF) + Gemini (OpenRouter). Adding Qwen/OLMo/Grok/GPT
targets back is a matter of extending ``config.TARGET_MODELS`` — the registry is
otherwise model-agnostic.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config_shim as cfg
from .base import ModelBackend
from .hf_backend import HFBackend
from .openrouter_backend import OpenRouterBackend


def build_backend(spec, *, adapter_path: str | None = None, **hf_kwargs) -> ModelBackend:
    if spec.backend == "hf":
        return HFBackend(
            spec.model_id, name=spec.name, kind=spec.kind,
            adapter_path=adapter_path, **hf_kwargs,
        )
    if spec.backend == "openrouter":
        return OpenRouterBackend(spec.model_id, name=spec.name)
    raise ValueError(f"Unknown backend {spec.backend!r} for {spec.name}")


@lru_cache(maxsize=None)
def get_backend(handle: str, adapter_path: str | None = None) -> ModelBackend:
    """Resolve a target-model handle to a (cached) backend instance."""
    if handle not in cfg.TARGET_MODELS:
        raise KeyError(
            f"{handle!r} not in scope. In-scope targets: {list(cfg.TARGET_MODELS)}"
        )
    return build_backend(cfg.TARGET_MODELS[handle], adapter_path=adapter_path)


@lru_cache(maxsize=1)
def get_judge_client():
    from .llm_client import AnthropicClient

    return AnthropicClient()


@lru_cache(maxsize=1)
def get_openrouter_client():
    from .llm_client import OpenAICompatClient

    return OpenAICompatClient()
