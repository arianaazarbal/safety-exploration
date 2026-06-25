"""Model factory: name -> live ModelClient. Caches instances so heavyweight
local Gemma weights load only once per process."""

from __future__ import annotations

from functools import lru_cache

import config

from .base import ModelClient


@lru_cache(maxsize=None)
def get_model(name: str, **backend_kwargs) -> ModelClient:
    if name not in config.MODELS:
        raise KeyError(
            f"Unknown model {name!r}. Known (Gemma/Gemini scope): "
            f"{sorted(config.MODELS)}"
        )
    spec = config.MODELS[name]
    if spec.backend == "hf":
        from .hf_model import HFModel
        return HFModel(spec, **backend_kwargs)
    if spec.backend == "openrouter":
        from .openrouter_model import OpenRouterModel
        return OpenRouterModel(spec)
    raise ValueError(f"Unsupported backend {spec.backend!r} for {name!r}")
