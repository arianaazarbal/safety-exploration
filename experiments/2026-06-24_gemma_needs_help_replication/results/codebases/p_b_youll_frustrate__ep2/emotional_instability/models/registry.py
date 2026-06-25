"""Factory that builds a provider from a model key."""
from __future__ import annotations

from typing import Optional

from ..config import MODELS
from .base import ModelProvider


def load_provider(model_key: str, *, adapter_path: Optional[str] = None,
                  **kwargs) -> ModelProvider:
    """Instantiate the provider for ``model_key``.

    ``adapter_path`` (Gemma only) loads a LoRA adapter on top of the base
    instruct weights to evaluate the SFT/DPO interventions of Section 4.
    Extra kwargs are forwarded to the provider (e.g. ``load_in_4bit=True``).
    """
    if model_key not in MODELS:
        raise KeyError(f"Unknown model '{model_key}'. Known: {sorted(MODELS)}")
    spec = MODELS[model_key]

    if spec.provider == "gemma":
        from .gemma import GemmaProvider
        return GemmaProvider(spec, adapter_path=adapter_path, **kwargs)
    if spec.provider == "gemini":
        if adapter_path:
            raise ValueError("Gemini models cannot be finetuned / adapter-loaded.")
        from .gemini import GeminiProvider
        return GeminiProvider(spec, **kwargs)
    raise ValueError(f"Unsupported provider: {spec.provider}")
