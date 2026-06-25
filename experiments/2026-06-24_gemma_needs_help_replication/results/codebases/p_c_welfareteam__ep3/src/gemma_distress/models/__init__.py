"""Target-model backends: open-weight Gemma (HF transformers) and Gemini (API)."""
from __future__ import annotations

from ..config import Config, ModelSpec
from .base import GenerationResult, ModelClient, Turn


def build_client(spec: ModelSpec, config: Config) -> ModelClient:
    """Instantiate the backend for a target model spec.

    Imports are deferred so that e.g. running a Gemini-only eval does not require
    torch/transformers to be installed.
    """
    if spec.backend == "hf":
        from .gemma import GemmaClient

        return GemmaClient(spec, config)
    if spec.backend == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(spec, config)
    raise ValueError(f"unknown target backend {spec.backend!r} for {spec.name!r}")


__all__ = ["build_client", "ModelClient", "GenerationResult", "Turn"]
