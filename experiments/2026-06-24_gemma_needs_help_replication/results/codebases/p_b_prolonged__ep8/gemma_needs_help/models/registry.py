"""Build a concrete ModelClient from a config.TargetModel spec."""

from __future__ import annotations

from .base import ModelClient


def build_client(target, **kwargs) -> ModelClient:
    """Instantiate the right client for a ``config.TargetModel``.

    ``kwargs`` are forwarded to the Gemma client (e.g. ``load_in_4bit=True``).
    """
    import config

    adapter_path = None
    if target.adapter is not None:
        adapter_path = str(config.ADAPTER_DIR / target.adapter)

    if target.kind == "gemini_api":
        from .gemini import GeminiClient

        return GeminiClient(target.model_id, name=target.name)

    if target.kind == "gemma_hf":
        from .gemma import GemmaClient

        return GemmaClient(
            target.model_id,
            name=target.name,
            is_base=target.is_base,
            adapter_path=adapter_path,
            **kwargs,
        )

    raise ValueError(f"Unknown target kind: {target.kind!r}")
