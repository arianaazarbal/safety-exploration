"""Map a ``ModelConfig`` to a concrete client, with per-process caching.

Caching matters for local Gemma: a 27B checkpoint should be loaded once and
reused across an entire sweep rather than per rollout.
"""
from __future__ import annotations

from ..config import Config, ModelConfig
from .api import AnthropicClient, OpenAIClient
from .base import ModelClient
from .gemini import GeminiClient
from .gemma import GemmaClient

_BACKENDS = {
    "gemma": GemmaClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
}

_CACHE: dict[str, ModelClient] = {}


def _cache_id(mc: ModelConfig) -> str:
    # adapter_path distinguishes finetuned variants of the same base model
    return f"{mc.key}:{mc.options.get('adapter_path', '')}"


def get_client(cfg: Config, key: str) -> ModelClient:
    mc = cfg.model(key)
    cid = _cache_id(mc)
    if cid in _CACHE:
        return _CACHE[cid]
    if mc.backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {mc.backend!r} for model {key!r}")
    client = _BACKENDS[mc.backend](
        model_id=mc.model_id,
        temperature=mc.temperature,
        max_tokens=mc.max_tokens,
        options=mc.options,
    )
    _CACHE[cid] = client
    return client


def clear_client_cache() -> None:
    _CACHE.clear()
