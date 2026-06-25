"""Common chat-model interface.

A ``Message`` is a simple ``{"role", "content"}`` dict, matching the OpenAI /
HF chat-template convention. Roles are ``"system"``, ``"user"``, ``"assistant"``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, TypedDict

from .. import config


class Message(TypedDict):
    role: str
    content: str


class ChatModel(Protocol):
    """Minimal interface every model client implements."""

    name: str

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        """Return a single assistant completion for ``messages``.

        If ``prefill`` is given, the assistant turn is *seeded* with that text
        and the model continues from it; the returned string is the
        continuation only (excluding the prefill). API models that cannot
        prefill raise :class:`NotImplementedError`.
        """
        ...

    def generate_batch(
        self,
        batch: Iterable[List[Message]],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> List[str]:
        ...


def build_model(model_key: str, runtime: Optional[config.RuntimeConfig] = None) -> ChatModel:
    """Factory dispatching on the model key.

    Gemma keys (``gemma-3-*``) build a local HF client; Gemini keys build an
    API client. Imports are lazy so that, e.g., running only Gemini evals does
    not require torch/transformers to be installed.
    """
    runtime = runtime or config.RUNTIME

    if model_key in config.GEMMA_INSTRUCT or model_key in config.GEMMA_BASE:
        from .hf_model import HFChatModel
        hf_id = (config.GEMMA_INSTRUCT | config.GEMMA_BASE)[model_key]
        is_base = model_key in config.GEMMA_BASE
        return HFChatModel(model_key, hf_id, is_base=is_base, runtime=runtime)

    if model_key in config.GEMINI_MODELS:
        from .api_model import APIChatModel
        return APIChatModel.for_gemini(model_key, runtime=runtime)

    raise ValueError(  # noqa: TRY003
        f"Unknown model key {model_key!r}. Known: "
        f"{sorted(config.GEMMA_INSTRUCT) + sorted(config.GEMMA_BASE) + sorted(config.GEMINI_MODELS)}"
    )


def build_finetuned_model(
    name: str,
    adapter_path: str,
    base_model_id: str = "google/gemma-3-27b-it",
    runtime: Optional[config.RuntimeConfig] = None,
) -> ChatModel:
    """Build a Gemma instruct model with a finetuned LoRA adapter (DPO/SFT)."""
    from .hf_model import HFChatModel
    return HFChatModel(name, base_model_id, is_base=False,
                       adapter_path=adapter_path, runtime=runtime or config.RUNTIME)
