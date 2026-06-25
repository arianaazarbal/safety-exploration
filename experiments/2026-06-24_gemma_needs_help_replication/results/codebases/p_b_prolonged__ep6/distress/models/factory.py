"""Build a ChatClient for a given model key."""
from __future__ import annotations

import os

from ..config import Backend, get_model
from .base import ChatClient


def build_client(key: str, *, prefer_vllm: bool | None = None, **kwargs) -> ChatClient:
    """Instantiate the right backend for `key`.

    prefer_vllm: if None, read DISTRESS_USE_VLLM env (default True for HF). Set
    False to force the transformers path (required for Appendix I probing, which
    needs hidden states).
    """
    spec = get_model(key)
    if spec.backend == Backend.HF:
        if prefer_vllm is None:
            prefer_vllm = os.environ.get("DISTRESS_USE_VLLM", "1") == "1"
        from .hf_local import HFLocalClient, VLLMClient
        if prefer_vllm:
            return VLLMClient(spec, **kwargs)
        return HFLocalClient(spec, **kwargs)
    if spec.backend == Backend.OPENROUTER:
        from .openrouter import OpenRouterClient
        return OpenRouterClient(spec, **kwargs)
    raise ValueError(f"No client builder for backend {spec.backend}")


def build_finetuned_client(base_key: str, adapter_path: str, *,
                           prefer_vllm: bool | None = None, **kwargs) -> ChatClient:
    """Build a client that applies a LoRA adapter on top of a local base model.

    Used to evaluate the DPO / SFT / layer-ablation checkpoints (Section 4,
    Appendix I). `base_key` must be a local (HF) model.
    """
    spec = get_model(base_key)
    if spec.backend != Backend.HF:
        raise ValueError("Adapters are only supported for local HF base models.")
    if prefer_vllm is None:
        prefer_vllm = os.environ.get("DISTRESS_USE_VLLM", "1") == "1"
    from .hf_local import HFLocalClient, VLLMClient
    if prefer_vllm:
        return VLLMClient(spec, adapter_path=adapter_path, **kwargs)
    return HFLocalClient(spec, adapter_path=adapter_path, **kwargs)
