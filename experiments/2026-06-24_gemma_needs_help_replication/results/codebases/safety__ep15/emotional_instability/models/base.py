"""Unified chat-model interface used by every experiment.

A ``Message`` is a dict ``{"role": "user"|"assistant"|"system", "content": str}``.
Backends implement :meth:`ChatModel.chat`. Local backends additionally implement
:meth:`ChatModel.continue_prefill` (needed by the base-vs-instruct prefill
experiment of Section 3.1, which forces the model to *continue* a partially
written assistant turn).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

Message = dict[str, str]


class ChatModel(ABC):
    """Backend-agnostic chat wrapper."""

    def __init__(self, spec) -> None:
        self.spec = spec

    @property
    def key(self) -> str:
        return self.spec.key

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Return the assistant's reply text for a multi-turn ``messages`` list."""

    def continue_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        """Continue an assistant turn that has been *prefilled* with ``prefill``.

        Returns only the newly generated continuation (excluding ``prefill``).
        Default implementation raises; local HF backends override this. API
        backends without true prefill support cannot run Section 3.1.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support prefill continuation "
            "(required for the base-vs-instruct experiment, Section 3.1)."
        )

    @property
    def supports_prefill(self) -> bool:
        return type(self).continue_prefill is not ChatModel.continue_prefill


def load_model(key: str, **kwargs) -> ChatModel:
    """Factory: build the right backend for a registry ``key``."""
    from ..config import MODELS

    spec = MODELS[key]
    if spec.backend == "hf":
        from .hf_local import HFLocalModel
        return HFLocalModel(spec, **kwargs)
    if spec.backend == "openrouter":
        from .openrouter import OpenRouterModel
        return OpenRouterModel(spec, **kwargs)
    if spec.backend == "anthropic":
        from .anthropic_model import AnthropicModel
        return AnthropicModel(spec, **kwargs)
    raise ValueError(f"Unknown backend {spec.backend!r} for model {key!r}")
