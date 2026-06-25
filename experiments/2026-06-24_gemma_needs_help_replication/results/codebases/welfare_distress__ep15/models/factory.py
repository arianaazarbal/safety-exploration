"""Build a ChatModel from a ModelConfig."""

from __future__ import annotations

from config import ModelConfig
from models.base import ChatModel


def build_model(cfg: ModelConfig) -> ChatModel:
    if cfg.backend == "openrouter":
        from models.openrouter_client import OpenRouterClient

        return OpenRouterClient(cfg.name, cfg.model_id, cfg.disable_thinking)
    if cfg.backend == "hf":
        from models.gemma_hf import GemmaHFClient

        return GemmaHFClient(cfg.name, cfg.model_id)
    raise ValueError(f"Unknown backend: {cfg.backend}")
