"""Build a ModelClient from a ModelSpec, picking the right backend."""

from __future__ import annotations

from ..config import ALL_MODELS, ModelSpec
from .base import ModelClient
from .hf_backend import HFModelClient
from .openrouter_backend import OpenRouterModelClient


def build_client(spec: ModelSpec, hf_backend: str = "vllm") -> ModelClient:
    if spec.backend == "hf":
        return HFModelClient(spec, backend=hf_backend)
    if spec.backend == "openrouter":
        return OpenRouterModelClient(spec)
    raise ValueError(f"unknown backend {spec.backend!r}")


def get_client(key: str, hf_backend: str = "vllm") -> ModelClient:
    return build_client(ALL_MODELS[key], hf_backend=hf_backend)
