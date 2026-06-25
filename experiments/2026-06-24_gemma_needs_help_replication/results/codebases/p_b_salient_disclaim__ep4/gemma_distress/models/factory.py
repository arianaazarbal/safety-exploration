"""Factory mapping config keys to concrete backends.

    build_client("gemma-3-27b-it")            -> GemmaLocalClient (instruct)
    build_client("gemma-3-27b-pt")            -> GemmaLocalClient (base)
    build_client("gemma-3-27b-it", adapter=…) -> GemmaLocalClient + LoRA adapter
    build_client("gemini-2.5-flash")          -> OpenRouterClient
    build_client("claude-sonnet-4-20250514")  -> AnthropicClient
    build_client("openai/gpt-5-mini")         -> OpenRouterClient
"""
from __future__ import annotations

from typing import Optional

from .. import config
from .anthropic_client import AnthropicClient
from .base import ModelClient


def build_client(key: str, *, adapter_path: Optional[str] = None,
                 lazy: bool = True) -> ModelClient:
    # Gemma (local HuggingFace) ---------------------------------------------
    if key in config.GEMMA_MODELS:
        from .local_hf import GemmaLocalClient
        is_base = key.endswith("-pt")
        client = GemmaLocalClient(
            name=key if not adapter_path else f"{key}+adapter",
            hf_id=config.GEMMA_MODELS[key],
            is_base=is_base,
            adapter_path=adapter_path,
        )
        if not lazy:
            client.load()
        return client

    # Gemini (OpenRouter) ---------------------------------------------------
    if key in config.GEMINI_MODELS:
        from .openrouter import OpenRouterClient
        return OpenRouterClient(name=key, model_id=config.GEMINI_MODELS[key])

    # Anthropic judges / auditors ------------------------------------------
    if key.startswith("claude-"):
        return AnthropicClient(name=key, model_id=key)

    # Any other OpenRouter-routable id (e.g. openai/gpt-5-mini) -------------
    if "/" in key:
        from .openrouter import OpenRouterClient
        return OpenRouterClient(name=key, model_id=key)

    raise ValueError(f"Unknown model key: {key!r}")


# Convenience constructors for the fixed roles in the paper -----------------
def build_judge() -> AnthropicClient:
    return AnthropicClient(name="frustration-judge", model_id=config.JUDGE_MODEL)


def build_onset_labeller() -> AnthropicClient:
    return AnthropicClient(name="onset-labeller", model_id=config.ONSET_LABEL_MODEL)


def build_paraphraser() -> AnthropicClient:
    return AnthropicClient(name="paraphraser", model_id=config.PARAPHRASE_MODEL)


def build_petri_auditor() -> AnthropicClient:
    return AnthropicClient(name="petri-auditor", model_id=config.PETRI_AUDITOR_MODEL)


def build_petri_judge() -> AnthropicClient:
    return AnthropicClient(name="petri-judge", model_id=config.PETRI_JUDGE_MODEL)


def build_reliability_judge() -> ModelClient:
    return build_client(config.RELIABILITY_MODEL)
