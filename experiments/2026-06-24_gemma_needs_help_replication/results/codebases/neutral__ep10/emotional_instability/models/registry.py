"""Construct a ChatModel from a name or a raw model id + backend."""

from __future__ import annotations

from typing import Optional

from .. import config
from .api_model import AnthropicModel, OpenRouterModel
from .base import ChatModel


def load_model(name: str, adapter_path: Optional[str] = None, **kwargs) -> ChatModel:
    """Load a registered model by its short name.

    `adapter_path` attaches a LoRA finetune (HF backend only) -- used to load
    the DPO / SFT models on top of Gemma-3-27b-it.
    """
    spec = config.get_model(name)
    if spec.backend == config.Backend.HF:
        from .hf_model import HFModel
        return HFModel(spec, adapter_path=adapter_path, **kwargs)
    if spec.backend == config.Backend.OPENROUTER:
        return OpenRouterModel(spec, **kwargs)
    if spec.backend == config.Backend.ANTHROPIC:
        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"Unsupported backend {spec.backend}")


def load_judge(model_id: str, backend: Optional[config.Backend] = None) -> ChatModel:
    """Load a judge / agent model by raw id (not in the MODELS registry).

    Routes Anthropic and GPT judges through the configured JUDGE_BACKEND
    (Anthropic API by default, OpenRouter if EMOINSTAB_JUDGE_BACKEND=openrouter).
    """
    backend = backend or config.JUDGE_BACKEND
    # Map a bare Anthropic id to the OpenRouter slug when routing via OpenRouter.
    if backend == config.Backend.OPENROUTER:
        slug = model_id
        if model_id.startswith("claude-"):
            slug = f"anthropic/{model_id}"
        elif model_id.startswith("gpt-"):
            slug = f"openai/{model_id}"
        spec = config.ModelSpec(model_id, slug, config.Backend.OPENROUTER, "judge")
        return OpenRouterModel(spec)
    if backend == config.Backend.ANTHROPIC:
        spec = config.ModelSpec(model_id, model_id, config.Backend.ANTHROPIC, "judge")
        return AnthropicModel(spec)
    raise ValueError(f"Unsupported judge backend {backend}")
