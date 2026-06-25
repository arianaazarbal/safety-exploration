"""Build (and cache) a `ModelClient` for a registered model name.

Local HF models are heavy, so clients are cached per-process. Finetuned Gemma
variants are created with `with_adapter(base_name, adapter_path)`.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from .. import config_proxy as cfg
from .base import ModelClient

_CACHE: dict[str, ModelClient] = {}


def _build(spec, *, backend_override: str | None = None,
           adapter_path: str | None = None) -> ModelClient:
    backend = backend_override or spec.backend
    if backend in ("openrouter", "openai"):
        from .api_client import OpenAICompatClient

        return OpenAICompatClient(spec.name, spec.model_id, backend=backend)
    if backend == "anthropic":
        from .api_client import AnthropicClient

        return AnthropicClient(spec.name, spec.model_id)
    if backend == "google":
        # Native Gemini backend; OpenRouter is the default path, so this is only
        # used when GOOGLE_API_KEY is set and the spec is switched over.
        from .google_client import GoogleGenAIClient

        return GoogleGenAIClient(spec.name, spec.model_id)
    if backend == "local_hf":
        from .local_client import LocalHFClient

        return LocalHFClient(
            spec.name, spec.model_id, is_base=spec.is_base,
            load_in_4bit=spec.load_in_4bit, adapter_path=adapter_path,
        )
    if backend == "vllm":
        from .vllm_client import VLLMClient

        return VLLMClient(spec.name, spec.model_id, adapter_path=adapter_path)
    raise ValueError(f"unknown backend {backend!r}")


def get_client(name: str, *, backend_override: str | None = None) -> ModelClient:
    cache_key = f"{name}::{backend_override or ''}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if name not in cfg.MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {list(cfg.MODELS)}")
    client = _build(cfg.MODELS[name], backend_override=backend_override)
    _CACHE[cache_key] = client
    return client


def with_adapter(base_name: str, adapter_path: str, *,
                 variant_name: str | None = None,
                 backend_override: str | None = None) -> ModelClient:
    """Load a LoRA adapter (DPO / SFT output) on top of a base instruct model.

    Registers the resulting variant under `variant_name` so its results files are
    labelled distinctly (e.g. 'gemma-3-27b-it-dpo')."""
    spec = cfg.MODELS[base_name]
    variant_name = variant_name or f"{base_name}-ft"
    spec = replace(spec, name=variant_name)
    client = _build(spec, backend_override=backend_override, adapter_path=adapter_path)
    _CACHE[f"{variant_name}::{backend_override or ''}"] = client
    return client
