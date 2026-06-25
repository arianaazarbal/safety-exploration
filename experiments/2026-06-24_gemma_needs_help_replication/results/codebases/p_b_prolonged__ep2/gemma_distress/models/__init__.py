"""Generation backends.

A ``TargetBackend`` produces responses from a *target* model under test
(Gemma via local HF, Gemini via OpenRouter). A ``JudgeBackend`` wraps the
Anthropic API for the judge / onset / paraphrase / Petri roles.

``get_target_backend(name, cfg)`` returns the right target backend for a model
key from ``config.MODELS``.
"""
from __future__ import annotations

from ..config import RunConfig, get_model
from .base import TargetBackend, ChatTurn


def get_target_backend(name: str, cfg: RunConfig) -> TargetBackend:
    spec = get_model(name)
    if spec.backend == "hf":
        from .hf_backend import HFBackend
        return HFBackend(spec, cfg)
    if spec.backend == "gemini":
        from .gemini_backend import GeminiBackend
        return GeminiBackend(spec, cfg)
    raise ValueError(f"Unknown backend '{spec.backend}' for model '{name}'")


__all__ = ["TargetBackend", "ChatTurn", "get_target_backend"]
