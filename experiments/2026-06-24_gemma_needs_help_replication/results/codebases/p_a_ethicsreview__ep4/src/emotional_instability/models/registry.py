"""Model registry: build a :class:`ChatModel` from ``config/models.yaml``."""

from __future__ import annotations

from functools import lru_cache

from ..utils.io import load_config
from .base import ChatModel
from .hf_backend import HFChatModel
from .openrouter_backend import OpenRouterChatModel


@lru_cache(maxsize=1)
def _registry() -> dict:
    return load_config("models")


def model_info(name: str) -> dict:
    reg = _registry()
    if name not in reg["models"]:
        raise KeyError(f"unknown model {name!r}; known: {list(reg['models'])}")
    return reg["models"][name]


def auxiliary_id(role: str) -> str:
    """Return the pinned model id for an auxiliary role (judge, auditor, ...)."""
    return _registry()["auxiliary"][role]


def decoding_defaults() -> dict:
    return _registry()["decoding"]


def load_model(name: str, adapter_path: str | None = None) -> ChatModel:
    info = model_info(name)
    backend = info["backend"]
    if backend == "hf":
        display = name if adapter_path is None else f"{name}+adapter"
        return HFChatModel(name=display, hf_id=info["hf_id"], role=info["role"],
                           adapter_path=adapter_path)
    if adapter_path is not None:
        raise ValueError(f"adapters are only supported for local HF models, not {name!r}")
    if backend == "openrouter":
        disable = decoding_defaults().get("disable_thinking", True)
        return OpenRouterChatModel(
            name=name, openrouter_id=info["openrouter_id"], disable_thinking=disable
        )
    raise ValueError(f"unknown backend {backend!r} for model {name!r}")
