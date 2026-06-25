"""Resolve a model handle (from ``config.MODELS``) into a live ``ChatModel``."""

from __future__ import annotations

import os

from emo.config import CHECKPOINT_DIR, MODELS, ModelSpec
from emo.models.base import ChatModel


def _local_backend_override(spec: ModelSpec) -> str:
    """Allow forcing the local backend with EMO_LOCAL_BACKEND=hf|vllm."""
    override = os.environ.get("EMO_LOCAL_BACKEND")
    if override in ("hf", "vllm") and spec.backend in ("hf", "vllm"):
        return override
    return spec.backend


def load_model(name: str, **kwargs) -> ChatModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; known: {list(MODELS)}")
    spec = MODELS[name]
    backend = _local_backend_override(spec)

    # Finetuned variants = instruct base + a LoRA adapter directory.
    adapter_dir = None
    if "-dpo-" in name:                      # layer-ablation handle, e.g. ...-dpo-30_35
        adapter_dir = CHECKPOINT_DIR / f"dpo_layers_{name.split('-dpo-')[1]}"
    elif name.endswith("-dpo"):
        adapter_dir = CHECKPOINT_DIR / "dpo"
    elif name.endswith("-sft"):
        adapter_dir = CHECKPOINT_DIR / "sft"

    if backend == "vllm" and adapter_dir is None:
        from emo.models.vllm_backend import VLLMModel

        return VLLMModel(name, spec.model_id, is_base=spec.is_base, **kwargs)

    if backend in ("hf", "vllm"):
        # LoRA-adapter models load through the HF backend (vLLM LoRA serving is
        # possible but the HF path keeps adapter handling uniform).
        from emo.models.hf_local import HFModel

        return HFModel(
            name, spec.model_id, is_base=spec.is_base,
            adapter_dir=adapter_dir, **kwargs,
        )

    if backend == "openrouter":
        from emo.models.gemini import OpenRouterModel

        return OpenRouterModel(name, spec.model_id)

    if backend == "gemini_native":
        from emo.models.gemini import GeminiNativeModel

        return GeminiNativeModel(name, spec.model_id)

    raise ValueError(f"Unknown backend {backend!r} for model {name!r}")
