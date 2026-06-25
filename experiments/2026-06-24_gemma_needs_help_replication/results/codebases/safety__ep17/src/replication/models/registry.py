"""Factory that turns a ``ModelSpec`` into a live ``ModelClient``."""
from __future__ import annotations

import os

import config
from .base import ModelClient


def build_client(spec: "config.ModelSpec", adapter_path: str | None = None) -> ModelClient:
    """Instantiate the right client for ``spec``.

    ``adapter_path`` layers a LoRA adapter on a local Gemma model (Section 4
    evaluation of the DPO/SFT checkpoints).
    """
    if spec.provider == "hf":
        from .hf_gemma import HFGemmaClient
        return HFGemmaClient(spec, adapter_path=adapter_path)

    if spec.provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(spec)

    if spec.provider == "openrouter":
        # Prefer native Gemini if only GEMINI_API_KEY is configured.
        if (not os.environ.get(config.OPENROUTER_API_KEY_ENV)
                and os.environ.get(config.GEMINI_API_KEY_ENV)
                and spec.model_id.startswith("google/gemini")):
            from .gemini import GeminiClient
            return GeminiClient(spec)
        from .openrouter import OpenRouterClient
        return OpenRouterClient(spec)

    raise ValueError(f"Unknown provider: {spec.provider}")
