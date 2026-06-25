"""Factory turning a ModelConfig (or a name + ModelRegistry) into a ChatModel."""

from __future__ import annotations

from ..config import ModelConfig, ModelRegistry
from .base import ChatModel


def build_model(cfg: ModelConfig, **hf_kwargs) -> ChatModel:
    if cfg.kind == "hf":
        from .hf import HFChatModel

        return HFChatModel(cfg, **hf_kwargs)
    if cfg.kind == "api":
        from .api import APIChatModel

        return APIChatModel(cfg)
    raise ValueError(f"Unknown model kind: {cfg.kind!r}")


def build_by_name(name: str, registry: ModelRegistry | None = None, **hf_kwargs) -> ChatModel:
    registry = registry or ModelRegistry()
    return build_model(registry.get(name), **hf_kwargs)
