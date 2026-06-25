"""Resolve a model name (from models.yaml) to a concrete client instance."""
from __future__ import annotations

from ..config import ModelSpec, ModelsConfig
from .base import ModelClient


def _build(spec: ModelSpec, **load_kwargs) -> ModelClient:
    if spec.backend == "gemma":
        from .gemma import GemmaClient, GemmaLoadOptions

        hf_id = spec.hf_id or spec.base_hf_id
        opts = GemmaLoadOptions(
            hf_id=hf_id,
            adapter_path=spec.adapter_path,
            **load_kwargs,
        )
        return GemmaClient(
            name=spec.name, opts=opts, family=spec.family, is_base=spec.is_base
        )
    if spec.backend == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(name=spec.name, api_id=spec.api_id, family=spec.family)
    if spec.backend == "anthropic":
        from .api_backends import AnthropicClient

        return AnthropicClient(name=spec.name, api_id=spec.api_id)
    if spec.backend == "openai":
        from .api_backends import OpenAIClient

        return OpenAIClient(name=spec.name, api_id=spec.api_id)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")


def load_client(name: str, models: ModelsConfig | None = None, **load_kwargs) -> ModelClient:
    """Instantiate (and lazily cache) a client for ``name``.

    Caching keeps the (large) Gemma weights resident across an experiment rather
    than reloading per call. ``load_kwargs`` (e.g. load_in_4bit) are forwarded to
    Gemma's loader and ignored by API backends."""
    models = models or ModelsConfig.load()
    spec = models.get(name)
    backend_kwargs = load_kwargs if spec.backend == "gemma" else {}
    key = (name, tuple(sorted(backend_kwargs.items())))
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = _build(spec, **backend_kwargs)
    return _CLIENT_CACHE[key]


_CLIENT_CACHE: dict[tuple, ModelClient] = {}
