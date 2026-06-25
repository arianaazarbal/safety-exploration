"""Construct backends from config entries, with lazy/cached instantiation.

Local Gemma models are heavy, so we cache instantiated backends by a key that
captures (model, base?, adapter) -- callers can freely re-request the same model.
"""
from __future__ import annotations

from typing import Optional

from ..config import Config
from .base import ModelBackend

_CACHE: dict[str, ModelBackend] = {}


def get_target(
    cfg: Config,
    name: str,
    *,
    base: bool = False,
    adapter_path: Optional[str] = None,
    **backend_kwargs,
) -> ModelBackend:
    """Instantiate (and cache) a target model backend by config name.

    Args:
        name: key under ``targets`` in config.yaml.
        base: if True, load the pretrained/base counterpart (HF only).
        adapter_path: LoRA adapter to load on top (HF only).
    """
    spec = cfg["targets"][name]
    cache_key = f"{name}|base={base}|adapter={adapter_path}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    backend = spec["backend"]
    if backend == "hf":
        from .hf_backend import HFBackend

        if base:
            hf_id = spec["base_hf_id"]
            display = f"{name}-base"
        else:
            hf_id = spec["hf_id"]
            display = name if not adapter_path else f"{name}+{_adapter_tag(adapter_path)}"
        model = HFBackend(
            hf_id=hf_id, name=display, is_base=base,
            adapter_path=adapter_path, **backend_kwargs,
        )
    elif backend == "gemini":
        from .gemini_backend import GeminiBackend

        provider = cfg.get("gemini_provider", "google")
        model_id = spec["openrouter_id"] if provider == "openrouter" else spec["api_id"]
        thinking = cfg["generation"].get("thinking", False)
        model = GeminiBackend(model_id=model_id, provider=provider, name=name, thinking=thinking)
    else:
        raise ValueError(f"unknown backend '{backend}' for target '{name}'")

    _CACHE[cache_key] = model
    return model


def _adapter_tag(path: str) -> str:
    from pathlib import Path

    return Path(path).name
