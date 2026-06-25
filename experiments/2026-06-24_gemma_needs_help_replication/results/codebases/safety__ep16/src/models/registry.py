"""Factory turning a registry name into a live model client."""

from __future__ import annotations

from config import MODEL_REGISTRY, ModelSpec
from src.models.api_model import OpenRouterChatModel
from src.models.base import ChatModel, CompletionModel
from src.models.hf_model import HFChatModel, HFCompletionModel


def get_chat_model(name: str, *, load_in_4bit: bool = False) -> ChatModel:
    spec = _spec(name)
    if spec.backend == "api":
        return OpenRouterChatModel(spec.name, spec.model_id)
    return HFChatModel(spec.name, spec.model_id, adapter=spec.adapter, load_in_4bit=load_in_4bit)


def get_completion_model(name: str, *, load_in_4bit: bool = False) -> CompletionModel:
    """Only valid for local HF base/instruct checkpoints (prefill experiment)."""
    spec = _spec(name)
    if spec.backend != "hf":
        raise ValueError(f"Completion (prefill) inference is only supported for HF models, not {name!r}.")
    return HFCompletionModel(spec.name, spec.model_id, load_in_4bit=load_in_4bit)


def _spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
