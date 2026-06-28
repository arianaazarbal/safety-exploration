"""Maps a provider name to a ModelClient. Extend here to add providers."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelClient

_PROVIDERS = {
    "anthropic": "moneypref.models.anthropic_client:AnthropicClient",
    "openai": "moneypref.models.openai_client:OpenAIClient",
    # To add Gemini: implement models/gemini_client.py and register it here, e.g.
    # "google": "moneypref.models.gemini_client:GeminiClient",
}


def build_client(spec: ModelSpec) -> ModelClient:
    if spec.provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {spec.provider!r}; known: {sorted(_PROVIDERS)}"
        )
    module_path, _, class_name = _PROVIDERS[spec.provider].partition(":")
    import importlib

    module = importlib.import_module(module_path)
    client_cls = getattr(module, class_name)
    return client_cls(spec.model_id)
