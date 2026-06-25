"""Factory that turns a :class:`ModelSpec` into a live :class:`ModelClient`.

Clients are cached by name so that loading a large local model (Gemma-27B) only
happens once per process. Backends are imported lazily inside ``build_client`` so
that, e.g., running an API-only experiment does not require torch/vLLM to be
installed.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import ModelRegistry, ModelSpec
from .base import ModelClient

_CLIENT_CACHE: dict[str, ModelClient] = {}


def build_client(spec: ModelSpec, registry: ModelRegistry | None = None) -> ModelClient:
    """Instantiate the client for ``spec`` (not cached; use :func:`get_client`)."""
    backend = spec.backend

    if backend == "vllm":
        from .vllm_client import VLLMClient

        # Finetuned specs reference a base target for the underlying weights.
        if spec.base is not None:
            assert registry is not None, "registry required to resolve finetuned base model"
            base_spec = registry.get(spec.base)
            hf_id = base_spec.hf_id
        else:
            hf_id = spec.hf_id
        return VLLMClient(
            spec.name,
            hf_id,
            adapter_path=spec.adapter_path,
            is_chat=spec.is_chat,
        )

    if backend == "hf":
        from .hf_client import HFClient

        return HFClient(spec.name, spec.hf_id, is_chat=spec.is_chat,
                        adapter_path=spec.adapter_path)

    if backend == "openrouter":
        from .openrouter_client import OpenRouterClient

        return OpenRouterClient(
            spec.name, spec.api_id, disable_thinking=spec.raw.get("disable_thinking", True)
        )

    if backend == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(spec.name, spec.api_id)

    if backend == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(spec.name, spec.api_id)

    raise ValueError(f"Unknown backend '{backend}' for model '{spec.name}'")


def get_client(name: str, registry: ModelRegistry | None = None) -> ModelClient:
    """Return a cached client for the named target/finetuned/role model."""
    if name in _CLIENT_CACHE:
        return _CLIENT_CACHE[name]
    registry = registry or ModelRegistry()
    spec = registry.get(name) if name in registry else registry.role(name)
    client = build_client(spec, registry)
    _CLIENT_CACHE[name] = client
    return client


@lru_cache(maxsize=1)
def default_registry() -> ModelRegistry:
    return ModelRegistry()
