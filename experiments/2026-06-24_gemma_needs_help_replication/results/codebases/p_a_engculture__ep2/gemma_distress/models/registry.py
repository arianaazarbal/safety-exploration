"""Model factory: build a :class:`ChatModel` from a :class:`ModelSpec`.

Local backends (``hf``, ``vllm``) are heavyweight (they load a 12–27B checkpoint onto the
GPU), so constructed models are cached per process and reused across experiment phases.
API backends are cheap and also cached for connection reuse.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import Config, ModelSpec
from .base import ChatModel

logger = logging.getLogger(__name__)

_CACHE: dict[str, ChatModel] = {}


def build_model(
    spec: ModelSpec,
    *,
    max_workers: int = 8,
    load_in_4bit: bool = False,
    cache: bool = True,
) -> ChatModel:
    """Instantiate the backend for ``spec``."""
    if cache and spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "hf":
        from .hf_backend import HFBackend

        model: ChatModel = HFBackend(
            spec.name,
            spec.model_id,
            is_base=spec.is_base,
            adapter_path=spec.extra.get("adapter"),
            load_in_4bit=load_in_4bit,
        )
    elif spec.backend == "vllm":
        from .vllm_backend import VLLMBackend

        model = VLLMBackend(
            spec.name,
            spec.model_id,
            is_base=spec.is_base,
            adapter_path=spec.extra.get("adapter"),
        )
    elif spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterBackend

        model = OpenRouterBackend(
            spec.name, spec.model_id, thinking=spec.thinking, max_workers=max_workers
        )
    elif spec.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        model = AnthropicBackend(spec.name, spec.model_id, max_workers=max_workers)
    else:
        raise ValueError(f"Unknown backend '{spec.backend}' for model '{spec.name}'.")

    if cache:
        _CACHE[spec.name] = model
    return model


def get_model(cfg: Config, name: str, **kwargs) -> ChatModel:
    """Build (or fetch cached) model by registry name."""
    return build_model(cfg.model(name), max_workers=cfg.eval.max_concurrency, **kwargs)


def clear_cache() -> None:
    """Release all cached models (frees GPU memory between phases)."""
    for model in _CACHE.values():
        try:
            model.close()
        except Exception:  # pragma: no cover
            pass
    _CACHE.clear()
