"""Backend interface shared by local (Gemma) and API (Gemini) models.

The rollout engine (``gnh.evaluation.rollout``) speaks only this interface, so
the same multi-turn protocol drives every model family identically -- the only
thing that differs is how a list of chat messages becomes a completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from gnh.config import MAX_NEW_TOKENS, TEMPERATURE, ModelSpec


@dataclass
class Message:
    """A single chat message. ``role`` is one of {"system", "user", "assistant"}."""

    role: str
    content: str


class ModelBackend(Protocol):
    """Anything that can turn a conversation into assistant text."""

    spec: ModelSpec

    def generate(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        prefill: str | None = None,
    ) -> list[str]:
        """Return ``n`` completions for ``messages``.

        ``prefill`` forces the assistant turn to *begin* with the given text;
        the returned strings EXCLUDE the prefill (only the continuation). Used
        by the §3 base-vs-instruct study. Backends that cannot prefill (closed
        APIs) raise :class:`NotImplementedError`.
        """
        ...


def get_backend(spec: ModelSpec, **kwargs) -> ModelBackend:
    """Instantiate the right backend for ``spec``.

    ``kwargs`` are forwarded to the backend constructor (e.g. ``adapter_path``
    for an HF LoRA finetune, or ``lora_layers`` for the Appendix-I ablation).
    """

    if spec.backend == "hf":
        from gnh.models.hf_backend import HFBackend

        # Fall back to the spec's default adapter (our §4 finetunes) unless the
        # caller overrides it explicitly.
        if spec.adapter_path and "adapter_path" not in kwargs:
            kwargs["adapter_path"] = spec.adapter_path
        return HFBackend(spec, **kwargs)
    if spec.backend == "openrouter":
        from gnh.models.openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r} for model {spec.key!r}")
