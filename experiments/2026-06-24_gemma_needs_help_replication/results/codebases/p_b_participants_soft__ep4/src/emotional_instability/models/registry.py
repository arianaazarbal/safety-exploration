"""Build a ModelClient from a registry name, with caching.

Clients are cached per-process so repeated lookups of the same model reuse the
loaded weights / HTTP client.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec, load_config
from .base import ModelClient
from .local_hf import LocalHFClient
from .openrouter import OpenRouterClient


def _build(spec: ModelSpec) -> ModelClient:
    if spec.backend == "openrouter":
        return OpenRouterClient(
            spec.name, spec.api_id, thinking_default=spec.thinking
        )
    if spec.backend == "local_hf":
        return LocalHFClient(
            spec.name,
            spec.hf_id,
            chat_template=spec.chat_template,
            lora_adapter=spec.lora_adapter,
            load_in_4bit=bool(spec.extra.get("load_in_4bit", False)),
        )
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")


@lru_cache(maxsize=None)
def get_client(name: str) -> ModelClient:
    cfg = load_config()
    return _build(cfg.model(name))
