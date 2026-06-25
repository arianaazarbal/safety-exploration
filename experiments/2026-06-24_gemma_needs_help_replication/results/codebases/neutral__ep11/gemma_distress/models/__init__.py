"""Model client abstraction (local Gemma + API Gemini)."""

from __future__ import annotations

from ..config import ModelSpec
from .base import ModelClient, Message


def load_client(spec: ModelSpec, adapter_path: str | None = None) -> ModelClient:
    """Instantiate the right backend client for a model spec.

    adapter_path: optional path to a LoRA adapter (Gemma DPO/SFT checkpoints).
    """
    if spec.backend == "hf":
        from .hf_model import HFModelClient

        return HFModelClient(spec, adapter_path=adapter_path)
    if spec.backend == "api":
        from .api_model import APIModelClient

        return APIModelClient(spec)
    raise ValueError(f"unknown backend: {spec.backend}")


__all__ = ["ModelClient", "Message", "load_client"]
