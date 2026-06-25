"""Factory that turns a ModelSpec name into a live ModelClient.

Local Gemma models are loaded lazily and cached so that the heavy 27B weights
are only materialised once per process.
"""

from __future__ import annotations

from config import CHECKPOINTS_DIR, MODELS, ModelSpec
from .base import ModelClient

_CACHE: dict[str, ModelClient] = {}


def load_model(name: str, *, load_in_4bit: bool | None = None) -> ModelClient:
    if name in _CACHE:
        return _CACHE[name]

    spec: ModelSpec = MODELS[name]
    if spec.backend == "openrouter":
        from .api_model import OpenRouterModel
        client: ModelClient = OpenRouterModel(spec.name, spec.model_id)

    elif spec.backend == "hf":
        from .hf_model import HFModel
        adapter_path = None
        kind = spec.kind
        if spec.kind == "finetune":
            # LoRA adapter produced by scripts/04_train_*.py
            adapter_path = str(CHECKPOINTS_DIR / spec.name)
            kind = "finetune"
        client = HFModel(
            spec.name, spec.model_id, kind=kind, adapter_path=adapter_path,
            load_in_4bit=spec.load_in_4bit if load_in_4bit is None else load_in_4bit,
        )
    else:
        raise ValueError(f"unknown backend {spec.backend}")

    _CACHE[name] = client
    return client


def unload_model(name: str) -> None:
    client = _CACHE.pop(name, None)
    if client is not None:
        client.close()
