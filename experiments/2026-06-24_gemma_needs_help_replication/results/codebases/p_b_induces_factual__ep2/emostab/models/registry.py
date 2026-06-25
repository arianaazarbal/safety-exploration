"""Build a `ChatModel` from a config name (+ optional LoRA adapter).

Local Gemma weights are heavy, so we memoise instances per (name, adapter) to
avoid reloading 27B weights when several experiments share a model in one run.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import Config
from .base import ChatModel


@lru_cache(maxsize=8)
def _build_cached(kind, name, hf_id, api_id, role, adapter_path, load_in_4bit):
    if kind == "hf":
        from .hf_model import HFModel

        return HFModel(
            name=name,
            hf_id=hf_id,
            role=role,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )
    if kind == "gemini":
        from .gemini_model import GeminiModel

        return GeminiModel(name=name, api_id=api_id, role=role)
    raise ValueError(f"unknown model kind: {kind!r}")


def build_model(
    cfg: Config,
    name: str,
    *,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
) -> ChatModel:
    spec = cfg.model_spec(name)
    return _build_cached(
        spec.kind,
        spec.name,
        spec.hf_id,
        spec.api_id,
        spec.role,
        adapter_path,
        load_in_4bit,
    )
