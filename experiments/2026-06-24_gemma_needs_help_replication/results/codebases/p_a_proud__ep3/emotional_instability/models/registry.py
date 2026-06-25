"""Factory that turns a :class:`ModelSpec` into a live :class:`ModelClient`.

Local (HF) models are cached per process — they are expensive to load and a GPU
typically holds one at a time, so callers are expected to evaluate a model fully
before requesting the next.
"""

from __future__ import annotations

from ..config import Config, ModelSpec
from .anthropic_client import AnthropicClient
from .base import ModelClient
from .hf_local import HFLocalClient
from .openrouter import OpenRouterClient

_CACHE: dict[str, ModelClient] = {}


def build_client(spec: ModelSpec) -> ModelClient:
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "hf_local":
        client: ModelClient = HFLocalClient(spec)
    elif spec.backend == "openrouter":
        client = OpenRouterClient(spec)
    elif spec.backend == "anthropic":
        client = AnthropicClient.from_spec(spec)
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.name!r}")

    _CACHE[spec.name] = client
    return client


def get_client(cfg: Config, name: str) -> ModelClient:
    return build_client(cfg.model(name))


def build_anthropic(model_id: str, name: str | None = None) -> AnthropicClient:
    """Build a Claude client by raw model id (judge / Petri roles)."""
    key = f"anthropic::{model_id}"
    if key not in _CACHE:
        _CACHE[key] = AnthropicClient(model_id, name=name)
    return _CACHE[key]  # type: ignore[return-value]


def clear_cache() -> None:
    _CACHE.clear()
