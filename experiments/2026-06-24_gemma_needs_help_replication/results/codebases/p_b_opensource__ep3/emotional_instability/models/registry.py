"""Construct :class:`ChatModel` instances from short keys or raw model IDs."""

from __future__ import annotations

import config
from config import ModelSpec

from .base import ChatModel


def get_spec(key: str) -> ModelSpec:
    if key in config.TARGET_MODELS:
        return config.TARGET_MODELS[key]
    raise KeyError(f"Unknown model key: {key!r}. "
                   f"Known: {sorted(config.TARGET_MODELS)}")


def build_model(key: str, *, adapter_path: str | None = None, **kwargs) -> ChatModel:
    """Build a target model by key.

    Finetuned Gemma variants are loaded as the 27B-it base plus a LoRA adapter;
    pass ``adapter_path`` (or use a ``gemma-3-27b-it-<variant>`` key together
    with the adapter directory your training run produced).
    """
    spec = get_spec(key)
    if spec.backend == "hf":
        from .hf_model import HFModel
        return HFModel(spec, adapter_path=adapter_path, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_model import AnthropicModel
        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"Unknown backend: {spec.backend!r}")


def build_judge(model_id: str, **kwargs) -> ChatModel:
    """Build a judge/auditor model from a raw provider model ID.

    Routing by prefix: ``claude-*`` → Anthropic API; ``gpt-*`` → OpenRouter
    (``openai/<id>``). This covers the paper's judges (Claude Sonnet/Opus) and
    the GPT-5-mini cross-validation judge.
    """
    if model_id.startswith("claude"):
        spec = ModelSpec(model_id, "anthropic", model_id, "claude", "instruct",
                         notes="judge/auditor")
        from .anthropic_model import AnthropicModel
        return AnthropicModel(spec, **kwargs)
    if model_id.startswith("gpt") or model_id.startswith("openai/"):
        router_id = model_id if "/" in model_id else f"openai/{model_id}"
        spec = ModelSpec(model_id, "openrouter", router_id, "gpt", "instruct",
                         notes="validation judge")
        from .openrouter import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    raise ValueError(f"Cannot infer backend for judge model id {model_id!r}")
