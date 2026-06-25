"""Build a :class:`ModelClient` from a :class:`ModelSpec`.

Backends are imported lazily so that, e.g., running only the API-based Gemini
evaluation does not require torch/vLLM to be installed, and vice versa.
"""
from __future__ import annotations

from functools import lru_cache

from emoinstab.config import ModelSpec, load_model_registry
from emoinstab.models.base import ModelClient


def build_client(spec: ModelSpec) -> ModelClient:
    backend = spec.backend
    if backend == "vllm":
        from emoinstab.models.local_vllm import VLLMClient

        return VLLMClient(spec)
    if backend == "transformers":
        from emoinstab.models.local_hf import HFClient

        return HFClient(spec)
    if backend == "gemini":
        from emoinstab.models.gemini import GeminiClient

        return GeminiClient(spec)
    if backend == "openrouter":
        from emoinstab.models.openrouter import OpenRouterClient

        return OpenRouterClient(spec)
    if backend == "anthropic":
        from emoinstab.models.anthropic_client import AnthropicClient

        return AnthropicClient(spec)
    if backend == "openai":
        from emoinstab.models.openai_client import OpenAIClient

        return OpenAIClient(spec)
    raise ValueError(f"Unknown backend: {backend!r}")


@lru_cache(maxsize=None)
def _registry() -> dict[str, ModelSpec]:
    return load_model_registry()


def get_client(name: str) -> ModelClient:
    """Look ``name`` up in configs/models.yaml and build its client."""
    reg = _registry()
    if name not in reg:
        raise KeyError(f"Model {name!r} not in registry. Known: {sorted(reg)}")
    return build_client(reg[name])
