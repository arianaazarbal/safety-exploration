"""Build a ChatModel from a ModelSpec or a registered key."""

from __future__ import annotations

from typing import Optional

from ..config import Backend, INSTRUMENTS, PARTICIPANTS, ModelSpec
from .base import ChatModel


def build_client(
    spec_or_key,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
) -> ChatModel:
    """Instantiate the right client for a participant ModelSpec/key.

    `adapter_path` attaches a LoRA adapter (the DPO/SFT Gemma variants).
    """
    spec: ModelSpec = (
        spec_or_key if isinstance(spec_or_key, ModelSpec) else PARTICIPANTS[spec_or_key]
    )

    if spec.backend is Backend.HF:
        from .hf_client import HFChatModel

        return HFChatModel(
            key=spec.key,
            model_id=spec.model_id,
            is_base=spec.is_base,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
        )

    if spec.backend is Backend.OPENROUTER:
        from .api_client import OpenRouterChatModel

        return OpenRouterChatModel(spec.key, spec.model_id, disable_thinking=True)

    raise ValueError(f"Unsupported participant backend: {spec.backend}")


def build_instrument(model_id: str) -> ChatModel:
    """Build a judge/auditor client by raw model id (dispatch on id prefix)."""
    if model_id.startswith("claude"):
        from .api_client import AnthropicChatModel

        return AnthropicChatModel(model_id, model_id)
    if model_id.startswith("gpt"):
        from .api_client import OpenAIChatModel

        return OpenAIChatModel(model_id, model_id)
    raise ValueError(f"Unknown instrument model id: {model_id}")
