"""Resolve model names to concrete clients.

Scope is restricted to Gemma + Gemini eval targets, the Claude judge/auditor
models, our finetuned Gemma adapters, and (optionally) the GPT-5-mini secondary
judge.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config
from .anthropic_client import AnthropicClient
from .base import ModelClient
from .huggingface_client import HuggingFaceClient
from .openrouter_client import OpenRouterClient

# Claude infrastructure models (judge/auditor) addressable by raw id.
_ANTHROPIC_IDS = {
    config.FRUSTRATION_JUDGE_MODEL,
    config.ONSET_LABEL_MODEL,
    config.PARAPHRASE_MODEL,
    config.PETRI_AUDITOR_MODEL,
    config.PETRI_JUDGE_MODEL,
}


@lru_cache(maxsize=None)
def get_client(name: str) -> ModelClient:
    """Return a (cached) client for `name`.

    `name` may be:
      * a target model key in config.TARGET_MODELS (gemma/gemini),
      * a finetuned model key in config.FINETUNED_MODELS,
      * a raw Anthropic model id (judge/auditor infrastructure),
      * the secondary judge ("gpt-5-mini").
    """
    if name in config.TARGET_MODELS:
        spec = config.TARGET_MODELS[name]
        if spec.backend == "huggingface":
            return HuggingFaceClient(
                spec.name, spec.model_id, chat_template=spec.chat_template)
        if spec.backend == "openrouter":
            return OpenRouterClient(
                spec.name, spec.model_id,
                disable_thinking=config.DISABLE_THINKING)
        raise ValueError(f"Unknown backend {spec.backend!r} for {name!r}")

    if name in config.FINETUNED_MODELS:
        ft = config.FINETUNED_MODELS[name]
        return HuggingFaceClient(
            name, ft["base"], chat_template=True, adapter_dir=ft["adapter"])

    if name in _ANTHROPIC_IDS or name.startswith("claude-"):
        return AnthropicClient(name, name)

    if name in ("gpt-5-mini", "openai/gpt-5-mini"):
        return OpenRouterClient("gpt-5-mini", "openai/gpt-5-mini")

    raise KeyError(
        f"Unknown model {name!r}. Known targets: "
        f"{sorted(config.TARGET_MODELS) + sorted(config.FINETUNED_MODELS)}")


def list_targets(*, include_base: bool = False,
                 include_finetuned: bool = False) -> list[str]:
    """List eval-target model names in scope."""
    targets = [
        n for n, s in config.TARGET_MODELS.items()
        if include_base or s.kind != "base"
    ]
    if include_finetuned:
        targets += list(config.FINETUNED_MODELS)
    return targets
