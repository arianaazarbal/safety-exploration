"""Build a :class:`ModelClient` from a :class:`config.ModelSpec`.

Clients are cached per (model handle, config id) so repeated lookups within a
run reuse the same loaded model / API session.
"""

from __future__ import annotations

from .base import ModelClient

_CLIENT_CACHE: dict[tuple, ModelClient] = {}


def get_client(spec, cfg) -> ModelClient:
    """Return a client for ``spec`` (a config.ModelSpec) under run config ``cfg``."""
    key = (spec.name, spec.backend, spec.model_id, spec.adapter_path, id(cfg))
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    if spec.backend == "hf":
        from .hf_client import HFModelClient

        client: ModelClient = HFModelClient(spec)
    elif spec.backend == "openrouter":
        from .api_client import OpenRouterClient

        client = OpenRouterClient(spec, cfg)
    elif spec.backend == "anthropic":
        from .api_client import AnthropicClient

        client = AnthropicClient(spec, cfg)
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")

    _CLIENT_CACHE[key] = client
    return client
