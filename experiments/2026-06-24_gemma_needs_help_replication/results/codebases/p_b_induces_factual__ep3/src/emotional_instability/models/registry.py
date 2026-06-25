"""Build :class:`ChatModel` instances from config entries.

Models are cached by name so a sweep that touches the same Gemma repeatedly
reuses the loaded weights. An optional ``adapter_path`` lets the same base
config key be instantiated as a LoRA-finetuned variant (used when evaluating
the SFT/DPO models against the vanilla instruct model).
"""

from __future__ import annotations

from functools import lru_cache

from ..config import Config
from .base import ChatModel
from .gemini import GeminiModel
from .hf_gemma import HFGemmaModel

_CACHE: dict[tuple[str, str | None], ChatModel] = {}


def build_model(name: str, cfg: Config, adapter_path: str | None = None) -> ChatModel:
    key = (name, adapter_path)
    if key in _CACHE:
        return _CACHE[key]

    spec = cfg.models[name]
    backend = spec.backend
    if backend == "hf_gemma":
        model: ChatModel = HFGemmaModel(
            name=name,
            hf_id=spec.hf_id,
            instruct=spec.get("instruct", True),
            adapter_path=adapter_path,
        )
    elif backend == "gemini":
        if adapter_path:
            raise ValueError("Gemini backend cannot load adapters")
        model = GeminiModel(name=name, api_id=spec.api_id)
    else:
        raise ValueError(f"unknown model backend {backend!r} for {name!r}")

    _CACHE[key] = model
    return model
