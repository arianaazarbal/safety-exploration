"""Resolve a model name from ``config.MODELS`` into a loaded ``ChatModel``.

Caches loaded backends so the same Gemma weights are not loaded twice within a
process. Gemma weights are heavy, so the caller is expected to evaluate one
local model fully before loading the next (the runner scripts do this).
"""

from __future__ import annotations

import config

from .base import ChatModel

_LOADED: dict[str, ChatModel] = {}


def load_model(name: str, *, adapter_path: str | None = None,
               prefer_vllm: bool = True) -> ChatModel:
    cache_key = f"{name}:{adapter_path}"
    if cache_key in _LOADED:
        return _LOADED[cache_key]

    spec = config.MODELS[name]
    if spec.backend == "gemma":
        from .gemma_backend import load_gemma

        model = load_gemma(
            spec.name, spec.hf_id, is_base=spec.is_base,
            adapter_path=adapter_path, prefer_vllm=prefer_vllm,
        )
    elif spec.backend == "gemini":
        from .gemini_backend import load_gemini

        model = load_gemini(spec.name, spec.hf_id)
    else:  # pragma: no cover
        raise ValueError(f"Unknown backend {spec.backend!r} for {name}")

    _LOADED[cache_key] = model
    return model


def unload(name: str, adapter_path: str | None = None) -> None:
    """Drop a loaded backend and free its resources (best effort)."""
    cache_key = f"{name}:{adapter_path}"
    model = _LOADED.pop(cache_key, None)
    if model is not None:
        try:
            model.close()
        except Exception:
            pass


def unload_all() -> None:
    for key in list(_LOADED):
        model = _LOADED.pop(key)
        try:
            model.close()
        except Exception:
            pass
