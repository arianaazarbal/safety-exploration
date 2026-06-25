"""Model loading by canonical name, with a small cache so repeated lookups in a
single process reuse the loaded weights."""
from __future__ import annotations

from functools import lru_cache

from .. import config
from .base import LLM
from .openrouter import OpenRouterModel


def get_spec(name: str) -> config.ModelSpec:
    if name not in config.MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Known: {sorted(config.MODELS)}")
    return config.MODELS[name]


@lru_cache(maxsize=None)
def load_model(name: str, load_in_4bit: bool = False) -> LLM:
    spec = get_spec(name)
    if spec.backend is config.Backend.OPENROUTER:
        return OpenRouterModel(
            name=spec.name,
            model_id=spec.model_id,
            is_instruct=spec.is_instruct,
            disable_thinking=config.DISABLE_THINKING and spec.supports_thinking_off,
        )
    elif spec.backend is config.Backend.HF:
        # Import lazily so API-only runs don't require torch/transformers.
        from .hf_local import HFModel
        return HFModel(
            name=spec.name,
            model_id=spec.model_id,
            is_instruct=spec.is_instruct,
            adapter_path=spec.adapter_path,
            load_in_4bit=load_in_4bit,
        )
    raise ValueError(f"Unhandled backend {spec.backend}")
