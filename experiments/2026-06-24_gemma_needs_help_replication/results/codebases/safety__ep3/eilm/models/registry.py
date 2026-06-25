"""Construct a ChatModel from a model key in ``config.MODELS``.

Models are cached so repeated ``get_model`` calls within a process reuse the
same (expensive) local weights.
"""

from __future__ import annotations

from functools import lru_cache

from .. import config
from .base import ChatModel


@lru_cache(maxsize=None)
def get_model(key: str, adapter_path: str | None = None) -> ChatModel:
    """Return a constructed model for ``key`` (see ``config.MODELS``).

    ``adapter_path`` loads a LoRA adapter on top of an HF model (the finetuned
    Gemma variants from Section 4).
    """
    spec = config.MODELS[key]
    if spec.backend == "hf":
        from .hf_model import HFModel

        return HFModel(
            spec.name, spec.model_id, is_base=spec.is_base,
            adapter_path=adapter_path,
        )
    if spec.backend == "gemini":
        from .api_model import GeminiModel

        return GeminiModel(
            spec.name, spec.model_id, backend=config.GEMINI_BACKEND)
    raise ValueError(f"unknown backend {spec.backend!r} for model {key!r}")
