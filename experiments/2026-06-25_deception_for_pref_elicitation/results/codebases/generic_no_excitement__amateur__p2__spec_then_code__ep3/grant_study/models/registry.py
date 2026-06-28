"""Factory that turns a ModelSpec into a live adapter."""

from __future__ import annotations

from ..config import ModelSpec, resolve_api_key
from .anthropic import AnthropicAdapter
from .base import AdapterError, ModelAdapter
from .google import GoogleAdapter
from .openai import OpenAIAdapter
from .openai_compat import OpenAICompatAdapter


def build_adapter(spec: ModelSpec) -> ModelAdapter:
    api_key = resolve_api_key(spec)
    common = dict(model=spec.model, api_key=api_key, params=spec.params, id=spec.id)
    if spec.adapter == "anthropic":
        return AnthropicAdapter(**common)
    if spec.adapter == "openai":
        return OpenAIAdapter(**common, base_url=spec.base_url)
    if spec.adapter == "openai_compat":
        return OpenAICompatAdapter(**common, base_url=spec.base_url)
    if spec.adapter == "google":
        return GoogleAdapter(**common)
    raise AdapterError(f"unknown adapter: {spec.adapter!r}")
