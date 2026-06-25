"""Factory mapping a registry name (or raw spec) to a concrete client.

Clients are cached so that repeated calls for the same model (e.g. across
conditions) reuse one loaded checkpoint -- important for the 27B Gemma, which is
expensive to load.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .. import config
from .base import ChatClient


@lru_cache(maxsize=None)
def get_client(name: str, *, adapter_path: Optional[str] = None,
               load_in_4bit: bool = False) -> ChatClient:
    """Instantiate (and cache) the client for a registered model name.

    ``adapter_path`` overrides the registry to attach a LoRA adapter on top of
    the named base model (used to evaluate DPO/SFT Gemma variants).
    """
    spec = config.MODELS.get(name)
    if spec is None:
        raise KeyError(f"Unknown model '{name}'. Known: {list(config.MODELS)}")

    backend = spec.backend
    effective_adapter = adapter_path or spec.adapter_path

    if backend in (config.BACKEND_HF, config.BACKEND_PEFT):
        from .gemma_hf import GemmaHFClient

        is_instruct = name.endswith("-it") or effective_adapter is not None
        return GemmaHFClient(
            spec.model_id, name,
            is_instruct=is_instruct,
            adapter_path=effective_adapter,
            load_in_4bit=load_in_4bit,
        )

    if backend == config.BACKEND_OPENROUTER:
        from .openrouter import OpenRouterClient

        return OpenRouterClient(spec.model_id, name)

    if backend == config.BACKEND_ANTHROPIC:
        from .anthropic_client import AnthropicClient

        return AnthropicClient(spec.model_id, name)

    raise ValueError(f"Unsupported backend '{backend}' for model '{name}'")


def get_anthropic(model_id: str):
    """Direct Anthropic client for the judge / Petri / onset / paraphrase roles."""
    from .anthropic_client import AnthropicClient

    return AnthropicClient(model_id)


def get_openrouter(model_id: str):
    """Direct OpenRouter client (e.g. GPT-5-mini judge-agreement validation)."""
    from .openrouter import OpenRouterClient

    return OpenRouterClient(model_id, model_id, disable_thinking=False)
