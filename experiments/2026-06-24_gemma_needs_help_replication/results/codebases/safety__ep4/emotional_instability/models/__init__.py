"""Inference backends for the replication.

Two backends behind one `ChatModel` interface:
  - hf_backend.GemmaHFModel       : local Gemma via vLLM (preferred) or transformers
  - openrouter_backend.OpenRouterModel : Gemini (and the GPT-5-mini judge) via API

`load_model(spec)` dispatches on `ModelSpec.backend`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from .base import ChatModel, ChatMessage  # noqa: E402


def load_model(spec: "config.ModelSpec", **kwargs) -> ChatModel:
    if spec.backend == "hf":
        from .hf_backend import GemmaHFModel
        return GemmaHFModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter_backend import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend}")


__all__ = ["ChatModel", "ChatMessage", "load_model"]
