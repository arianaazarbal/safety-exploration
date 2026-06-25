"""Build a participant ``ChatModel`` from its config name.

Resolves a model name (e.g. ``gemini-2.5-flash`` or ``gemma-3-27b-it``) against
the ``models`` block of the config and instantiates the right backend. HF models
are loaded lazily (heavy import) only when requested.
"""
from __future__ import annotations

from typing import Optional

from .base import ChatModel


def _find_spec(cfg: dict, name: str) -> dict:
    for family in ("gemma", "gemini"):
        block = cfg["models"].get(family, {})
        if name in block:
            return block[name]
    raise KeyError(f"model {name!r} not found under models.gemma / models.gemini in config")


def build_model(cfg: dict, name: str, adapter_path: Optional[str] = None) -> ChatModel:
    spec = _find_spec(cfg, name)
    backend = spec["backend"]
    if backend == "openrouter":
        from .openrouter import OpenRouterModel

        return OpenRouterModel(name=name, openrouter_id=spec["openrouter_id"],
                               thinking=spec.get("thinking", False))
    if backend == "hf":
        from .hf_gemma import HFGemmaModel

        return HFGemmaModel(
            name=name,
            hf_id=spec["hf_id"],
            dtype=spec.get("dtype", "bfloat16"),
            load_in_4bit=spec.get("load_in_4bit", True),
            is_base=spec["hf_id"].endswith("-pt"),
            adapter_path=adapter_path,
        )
    raise ValueError(f"unknown backend {backend!r} for model {name!r}")
