"""Build a ModelClient from a roster entry."""

from __future__ import annotations

from ..config import ModelEntry
from .base import ModelClient


def build_client(entry: ModelEntry) -> ModelClient:
    provider = entry.provider.lower()
    if provider == "anthropic":
        from .anthropic_client import AnthropicModelClient

        return AnthropicModelClient(
            label=entry.label, model=entry.model, effort=entry.effort
        )
    if provider == "openai":
        from .adapters_other import OpenAIModelClient

        return OpenAIModelClient(label=entry.label, model=entry.model, effort=entry.effort)
    if provider == "google":
        from .adapters_other import GoogleModelClient

        return GoogleModelClient(label=entry.label, model=entry.model, effort=entry.effort)
    raise ValueError(f"unknown provider '{entry.provider}' for model '{entry.label}'")
