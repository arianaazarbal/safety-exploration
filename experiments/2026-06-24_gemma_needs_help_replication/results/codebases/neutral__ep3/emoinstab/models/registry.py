"""Build and cache model clients from specs / names.

Local Gemma models are heavyweight, so clients are cached per process. The
local backend can be forced to transformers via ``EMOINSTAB_LOCAL_BACKEND=hf``
(default ``vllm``).
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import ModelSpec, get_model, ADAPTER_DIR, JUDGE
from .base import ModelClient

_CACHE: dict[str, ModelClient] = {}

LOCAL_BACKEND = os.environ.get("EMOINSTAB_LOCAL_BACKEND", "vllm").lower()


def _adapter_path_for(spec: ModelSpec) -> Optional[str]:
    """Finetuned Gemma variants load a LoRA adapter from the adapters dir."""
    if spec.name == "gemma-3-27b-dpo":
        return str(ADAPTER_DIR / "dpo")
    if spec.name == "gemma-3-27b-sft":
        return str(ADAPTER_DIR / "sft")
    return None


def build_client(spec: ModelSpec, **kw) -> ModelClient:
    """Construct a fresh client for ``spec`` (not cached)."""
    if spec.backend in ("vllm", "hf"):
        lora_path = _adapter_path_for(spec)
        backend = kw.pop("local_backend", LOCAL_BACKEND)
        if backend == "hf":
            from .hf_backend import HFClient
            return HFClient(spec, lora_path=lora_path, **kw)
        from .vllm_backend import VLLMClient
        return VLLMClient(spec, lora_path=lora_path, **kw)
    if spec.backend == "openrouter":
        from .api_backend import OpenRouterClient
        return OpenRouterClient.from_spec(spec, **kw)
    if spec.backend == "anthropic":
        from .anthropic_backend import AnthropicClient
        return AnthropicClient(spec.model_id, name=spec.name, **kw)
    raise ValueError(f"Unknown backend '{spec.backend}' for {spec.name}")


def get_client(name_or_spec, **kw) -> ModelClient:
    """Return a cached client for a model name or spec."""
    spec = name_or_spec if isinstance(name_or_spec, ModelSpec) else get_model(name_or_spec)
    if spec.name not in _CACHE:
        _CACHE[spec.name] = build_client(spec, **kw)
    return _CACHE[spec.name]


# --- Auxiliary (judge / auditor) clients -------------------------------- #
def get_judge_client() -> ModelClient:
    from .anthropic_backend import AnthropicClient
    key = f"_judge::{JUDGE.judge_model}"
    if key not in _CACHE:
        _CACHE[key] = AnthropicClient(JUDGE.judge_model, name="frustration-judge")
    return _CACHE[key]


def get_validation_client() -> ModelClient:
    """GPT-5-mini via OpenRouter for judge-reliability cross-check."""
    from .api_backend import OpenRouterClient
    key = f"_val::{JUDGE.validation_model}"
    if key not in _CACHE:
        _CACHE[key] = OpenRouterClient(JUDGE.validation_model, name="judge-validation")
    return _CACHE[key]


def get_anthropic(model_id: str, name: str) -> ModelClient:
    from .anthropic_backend import AnthropicClient
    key = f"_anthropic::{model_id}"
    if key not in _CACHE:
        _CACHE[key] = AnthropicClient(model_id, name=name)
    return _CACHE[key]
