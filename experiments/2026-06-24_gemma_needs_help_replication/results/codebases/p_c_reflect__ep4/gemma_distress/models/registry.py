"""Factory for target-model clients, scoped to Gemma + Gemini."""

from __future__ import annotations

from gemma_distress import config
from gemma_distress.models.base import ModelClient


def load_client(
    spec_or_name: "config.ModelSpec | str",
    *,
    adapter_path: str | None = None,
    use_openrouter: bool = False,
    **kwargs,
) -> ModelClient:
    """Instantiate the right client for a model spec.

    Parameters
    ----------
    spec_or_name:
        A :class:`config.ModelSpec` or the short model name (e.g.
        ``"gemma-3-27b-it"``).
    adapter_path:
        Optional PEFT/LoRA adapter directory (Gemma only; used to load the
        finetuned models from Section 4).
    use_openrouter:
        Route Gemini through OpenRouter instead of the native GenAI SDK.
    """
    spec = config.ALL_MODELS[spec_or_name] if isinstance(spec_or_name, str) else spec_or_name

    if spec.backend == "gemma":
        from gemma_distress.models.gemma import GemmaClient

        return GemmaClient(spec, adapter_path=adapter_path, **kwargs)
    if spec.backend == "gemini":
        if adapter_path:
            raise ValueError("Gemini models cannot take a local adapter.")
        from gemma_distress.models.gemini import GeminiClient

        return GeminiClient(spec, use_openrouter=use_openrouter, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.name!r}")
