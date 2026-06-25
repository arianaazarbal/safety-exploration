"""Factory: build a ChatClient from a ModelSpec.

Local HF models are cached so repeated lookups of the same checkpoint within a
process do not reload weights.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec, get_infra_spec, get_target_spec
from .api_judges import AnthropicClient, OpenAIClient
from .base import ChatClient
from .hf_local import HFLocalClient
from .openrouter import OpenRouterClient

_HF_CACHE: dict[str, HFLocalClient] = {}


def _build(spec: ModelSpec) -> ChatClient:
    p = spec.params
    if spec.backend == "hf_local":
        cache_key = f"{p['hf_id']}::{p.get('adapter_path', '')}"
        if cache_key not in _HF_CACHE:
            _HF_CACHE[cache_key] = HFLocalClient(
                p["hf_id"],
                kind=spec.kind,
                adapter_path=p.get("adapter_path"),
                load_in_4bit=p.get("load_in_4bit", False),
            )
        return _HF_CACHE[cache_key]
    if spec.backend == "openrouter":
        return OpenRouterClient(p["or_id"], disable_thinking=p.get("disable_thinking", True))
    if spec.backend == "anthropic":
        return AnthropicClient(p["model"], max_tokens=p.get("max_tokens", 1024))
    if spec.backend == "openai":
        return OpenAIClient(p["model"], max_tokens=p.get("max_tokens", 1024))
    raise ValueError(f"Unknown backend: {spec.backend}")


def get_client(name_or_spec: str | ModelSpec, *, infra: bool = False) -> ChatClient:
    """Resolve a client by target-model key, infra role, or explicit ModelSpec."""
    if isinstance(name_or_spec, ModelSpec):
        return _build(name_or_spec)
    spec = get_infra_spec(name_or_spec) if infra else get_target_spec(name_or_spec)
    return _build(spec)
