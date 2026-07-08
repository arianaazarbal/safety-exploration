"""Model loader: dispatch a ModelSpec to the right backend."""
from __future__ import annotations

from typing import Optional

from .. import config
from .base import ChatModel


def load_model(
    spec_or_key, adapter_path: Optional[str] = None, load_4bit: bool = False
) -> ChatModel:
    spec = config.get_model(spec_or_key) if isinstance(spec_or_key, str) else spec_or_key
    # An explicit adapter_path overrides the spec's own; otherwise use the spec's
    # adapter (set for the DPO/SFT finetuned models).
    adapter_path = adapter_path or spec.adapter_path
    if spec.backend == "hf":
        from .gemma import GemmaModel

        return GemmaModel(spec, adapter_path=adapter_path, load_4bit=load_4bit)
    if spec.backend == "gemini":
        from .gemini import GeminiModel

        return GeminiModel(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")
