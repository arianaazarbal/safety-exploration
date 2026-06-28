"""Resolve a ``provider:model`` reference string into a :class:`ModelClient`."""

from __future__ import annotations

from typing import Any

from .base import ModelClient

# Default model id per provider when the ref omits one ("anthropic" -> default).
_DEFAULTS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5",
    "google": "gemini-2.5-pro",
}


def parse_model_ref(ref: str) -> tuple[str, str]:
    """``"anthropic:claude-opus-4-8"`` -> ``("anthropic", "claude-opus-4-8")``."""
    if ":" in ref:
        provider, model_id = ref.split(":", 1)
    else:
        provider, model_id = ref, ""
    provider = provider.strip().lower()
    model_id = model_id.strip() or _DEFAULTS.get(provider, "")
    if not model_id:
        raise ValueError(f"Unknown provider with no default model: {provider!r}")
    return provider, model_id


def build_client(ref: str, **kwargs: Any) -> ModelClient:
    """Construct the adapter for ``ref``. Adapters import their SDK lazily."""
    provider, model_id = parse_model_ref(ref)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        client = AnthropicClient(model_id, **kwargs)
    elif provider == "openai":
        from .openai_client import OpenAIClient

        client = OpenAIClient(model_id, **kwargs)
    elif provider == "google":
        from .google_client import GoogleClient

        client = GoogleClient(model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")
    client.model_ref = ref
    return client
