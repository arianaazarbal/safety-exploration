"""Construct a ``ModelClient`` from a ``ModelSpec``.

Backends are instantiated lazily and cached, because loading a 27B model into
vLLM/transformers is expensive and we only want to pay for it once per process.
The judge and Petri models are constructed separately from their config blocks.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import ModelSpec
from .anthropic_client import AnthropicClient
from .base import ModelClient
from .openrouter import OpenRouterClient

# vLLM / HF backends are imported lazily inside ``build_client`` so the package
# stays importable (and the unit tests run) without a GPU/vLLM/torch install.

# Cache keyed by the immutable spec; value is the live client.
_CACHE: dict[str, ModelClient] = {}


def build_client(spec: ModelSpec) -> ModelClient:
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "openrouter":
        assert spec.api_id, f"{spec.name}: openrouter backend needs api_id"
        client: ModelClient = OpenRouterClient(spec.name, spec.api_id)
    elif spec.backend == "anthropic":
        assert spec.api_id, f"{spec.name}: anthropic backend needs api_id"
        client = AnthropicClient(spec.name, spec.api_id)
    elif spec.backend == "vllm":
        from .vllm_backend import VLLMClient

        assert spec.hf_id, f"{spec.name}: vllm backend needs hf_id"
        client = VLLMClient(spec.name, spec.hf_id, adapter_path=spec.adapter_path)
    elif spec.backend == "hf":
        from .hf_local import HFLocalClient

        assert spec.hf_id, f"{spec.name}: hf backend needs hf_id"
        client = HFLocalClient(
            spec.name, spec.hf_id, adapter_path=spec.adapter_path, is_base=spec.is_base
        )
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'")

    _CACHE[spec.name] = client
    return client


@lru_cache(maxsize=8)
def build_judge(model_id: str) -> AnthropicClient:
    """Build the Claude judge / Petri judge. Cached by model id."""
    return AnthropicClient(f"judge:{model_id}", model_id)
