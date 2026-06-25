"""Build a ChatModel instance from a registry name (or a finetuned adapter)."""
from __future__ import annotations

from typing import Optional

from ..config import MODELS, Backend, ModelSpec
from .base import ChatModel
from .hf_local import HFLocalModel
from .openrouter import OpenRouterModel


def build_model(name: str, load_in_4bit: bool = False) -> ChatModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODELS)}")
    spec: ModelSpec = MODELS[name]
    if spec.backend in (Backend.HF_INSTRUCT, Backend.HF_BASE):
        return HFLocalModel(
            name=spec.name,
            model_id=spec.model_id,
            is_instruct=spec.is_instruct,
            load_in_4bit=load_in_4bit,
        )
    if spec.backend == Backend.OPENROUTER:
        return OpenRouterModel(name=spec.name, model_id=spec.model_id)
    raise ValueError(f"Unsupported backend {spec.backend}")


def load_finetuned(
    name: str,
    adapter_dir: str,
    base_model: str = "gemma-3-27b-it",
    load_in_4bit: bool = False,
) -> ChatModel:
    """Load a LoRA-finetuned Gemma (DPO/SFT) by attaching `adapter_dir` to the
    instruct base weights."""
    spec = MODELS[base_model]
    return HFLocalModel(
        name=name,
        model_id=spec.model_id,
        is_instruct=True,
        adapter_dir=adapter_dir,
        load_in_4bit=load_in_4bit,
    )
