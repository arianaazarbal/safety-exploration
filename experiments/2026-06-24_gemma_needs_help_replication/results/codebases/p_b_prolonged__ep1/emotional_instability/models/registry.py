"""Factory that resolves a model name to a concrete ``ChatModel``.

Accepts:
* target model keys from ``config.TARGET_MODELS`` (Gemma / Gemini),
* finetuned Gemma variants from ``config.FINETUNED_VARIANTS`` (LoRA adapter on
  the instruct base),
* bare provider ids for the measurement models (Claude / GPT / Gemini), used by
  the judge, Petri, onset/paraphrase helpers.
"""

from __future__ import annotations

from typing import Optional

import config
from .base import ChatModel


def available_models() -> list[str]:
    return list(config.TARGET_MODELS) + list(config.FINETUNED_VARIANTS)


def build_model(name: str, use_vllm: bool = False, **kwargs) -> ChatModel:
    # ---- finetuned Gemma variants (base instruct + LoRA adapter) ----
    if name in config.FINETUNED_VARIANTS:
        from .hf_backend import HFChatModel

        spec_name = config.FINETUNED_VARIANTS[name]["base"]
        base = config.TARGET_MODELS[spec_name]
        adapter = str(config.FINETUNED_VARIANTS[name]["adapter"])
        return HFChatModel(
            name=name,
            model_id=base.model_id,
            is_base=base.is_base,
            adapter_path=adapter,
            use_vllm=use_vllm,
            **kwargs,
        )

    # ---- target models ----
    if name in config.TARGET_MODELS:
        spec = config.TARGET_MODELS[name]
        return _build_from_spec(spec, use_vllm=use_vllm, **kwargs)

    # ---- infrastructure / arbitrary provider ids ----
    return _build_infrastructure(name, **kwargs)


def _build_from_spec(spec: config.ModelSpec, use_vllm: bool = False, **kwargs) -> ChatModel:
    if spec.backend == "hf":
        from .hf_backend import HFChatModel

        return HFChatModel(
            name=spec.name,
            model_id=spec.model_id,
            is_base=spec.is_base,
            use_vllm=use_vllm,
            **kwargs,
        )
    if spec.backend == "gemini":
        from .gemini_backend import GeminiChatModel

        return GeminiChatModel(name=spec.name, model_id=spec.model_id, **kwargs)
    raise ValueError(f"Unknown backend: {spec.backend}")


def _build_infrastructure(model_id: str, **kwargs) -> ChatModel:
    """Resolve a raw provider id for judge / auditor / validation roles."""
    lower = model_id.lower()
    if lower.startswith("claude"):
        from .anthropic_backend import AnthropicChatModel

        return AnthropicChatModel(name=model_id, model_id=model_id, **kwargs)
    if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
        from .openai_backend import OpenAIChatModel

        return OpenAIChatModel(name=model_id, model_id=model_id, **kwargs)
    if lower.startswith("gemini"):
        from .gemini_backend import GeminiChatModel

        return GeminiChatModel(name=model_id, model_id=model_id, **kwargs)
    raise ValueError(f"Cannot resolve a backend for model id: {model_id!r}")
