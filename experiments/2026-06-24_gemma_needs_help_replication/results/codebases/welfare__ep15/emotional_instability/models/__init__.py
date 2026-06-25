"""Model backend factory."""

from __future__ import annotations

import config
from .base import Message, ModelBackend
from .hf_backend import HFBackend
from .openrouter_backend import OpenRouterBackend

__all__ = ["Message", "ModelBackend", "HFBackend", "OpenRouterBackend", "build_model"]


def build_model(name: str, lora_path: str | None = None, **kwargs) -> ModelBackend:
    """Construct a backend by short model name.

    Recognised names: any key of config.GEMMA_INSTRUCT / GEMMA_BASE /
    GEMINI_OPENROUTER, or "dpo-gemma-3-27b" / "sft-gemma-3-27b" for fine-tunes
    (pass the adapter directory via `lora_path`).
    """
    if name in config.GEMINI_OPENROUTER:
        return OpenRouterBackend(name, config.GEMINI_OPENROUTER[name],
                                 concurrency=config.OPENROUTER_CONCURRENCY)
    if name in config.GEMMA_INSTRUCT:
        return HFBackend(name, config.GEMMA_INSTRUCT[name], is_chat=True,
                         lora_path=lora_path, **kwargs)
    if name in config.GEMMA_BASE:
        return HFBackend(name, config.GEMMA_BASE[name], is_chat=False,
                         lora_path=lora_path, **kwargs)
    if name.startswith(("dpo-", "sft-")):
        # Fine-tuned instruct model: load the 27B-it base with a LoRA adapter.
        if lora_path is None:
            raise ValueError(f"{name} requires lora_path to the trained adapter")
        return HFBackend(name, config.DPO_BASE_MODEL, is_chat=True,
                         lora_path=lora_path, **kwargs)
    raise ValueError(f"Unknown model name: {name!r}")
