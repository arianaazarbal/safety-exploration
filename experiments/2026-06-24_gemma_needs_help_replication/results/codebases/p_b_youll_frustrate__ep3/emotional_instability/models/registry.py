"""Resolve a model name to a concrete :class:`ModelClient`.

Routing rule (matches the paper's setup):
* ``google/gemini-*``  -> OpenRouter (API).
* ``google/gemma-*``   -> local HuggingFace weights.
* anything else        -> OpenRouter (lets the same code drive judges / extra
                          baselines if desired).
"""

from __future__ import annotations

from typing import Optional

from .. import config
from .base import ModelClient
from .gemma import GemmaClient
from .openrouter import OpenRouterClient


def build_client(
    name: str,
    settings: Optional[config.Settings] = None,
    adapter_path: Optional[str] = None,
    **kwargs,
) -> ModelClient:
    settings = settings or config.DEFAULT
    lname = name.lower()

    if "gemma" in lname:
        return GemmaClient(name, adapter_path=adapter_path, **kwargs)

    if "gemini" in lname or name.startswith(("anthropic/", "openai/", "x-ai/")):
        return OpenRouterClient(
            name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            **kwargs,
        )

    # Default: assume an OpenRouter-routable id.
    return OpenRouterClient(
        name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        **kwargs,
    )
