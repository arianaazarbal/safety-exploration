"""Model factory. Resolves a registry key to a concrete ``ChatModel`` (target) or a
text-completion client (judge / auxiliary).

Fine-tuned Gemma variants use the key convention ``<base-key>+<adapter-name>`` where
``<adapter-name>`` is a sub-directory under ``CHECKPOINT_DIR`` containing a LoRA adapter,
e.g. ``gemma-3-27b-it+dpo``. This lets the eval harness treat the DPO/SFT models exactly
like any other target.
"""
from __future__ import annotations

from functools import lru_cache

import config
from .base import ChatModel
from .gemma import GemmaModel
from .gemini import GeminiModel
from .anthropic_client import AnthropicClient
from .openai_client import OpenAIClient


def _resolve_target_spec(key: str):
    """Return (ModelSpec, adapter_path_or_None) for a possibly-adapter-suffixed key."""
    if "+" in key:
        base_key, adapter = key.split("+", 1)
        spec = config.TARGET_MODELS[base_key]
        adapter_path = str(config.CHECKPOINT_DIR / adapter)
        return spec, adapter_path
    return config.TARGET_MODELS[key], None


@lru_cache(maxsize=None)
def build_model(key: str, *, load_in_4bit: bool = False) -> ChatModel:
    """Build (and cache) a target chat model by registry key."""
    spec, adapter_path = _resolve_target_spec(key)
    if spec.backend == "gemma_hf":
        return GemmaModel(
            name=key,
            model_id=spec.model_id,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )
    if spec.backend == "gemini":
        return GeminiModel(name=key, model_id=spec.model_id)
    raise ValueError(f"Unknown backend {spec.backend!r} for target {key!r}")


@lru_cache(maxsize=None)
def get_text_completion_client(model_id: str):
    """Build a completion client for a judge/auxiliary model id.

    Backend is inferred from the id prefix: ``claude*`` -> Anthropic, ``gpt*`` -> OpenAI.
    """
    if model_id.startswith("claude"):
        return AnthropicClient(model_id)
    if model_id.startswith("gpt") or model_id.startswith("o"):
        return OpenAIClient(model_id)
    raise ValueError(f"Cannot infer backend for completion model {model_id!r}")
