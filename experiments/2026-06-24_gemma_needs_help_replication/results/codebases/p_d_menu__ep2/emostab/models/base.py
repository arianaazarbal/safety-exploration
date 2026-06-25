"""Abstract model-client interface shared by all backends."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationResult:
    text: str
    # Prefill (if the response was seeded with a partial assistant turn) is
    # excluded from `text` and stored here so callers can score continuations
    # alone (Section 3).
    prefill: str = ""
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


class ModelClient(abc.ABC):
    """Common surface for subject + infrastructure models.

    Not all backends implement every method; capability flags advertise support.
    """

    supports_prefill: bool = False        # seed an assistant turn and continue it
    supports_logits: bool = False         # residual-stream / logit capture (App I)
    is_base_model: bool = False           # pretrained (no chat template)

    def __init__(self, spec):
        self.spec = spec
        self.is_base_model = spec.is_base

    # --- core generation --------------------------------------------------- #
    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate one assistant turn given a chat history.

        If `prefill` is provided and `supports_prefill`, the assistant turn is
        seeded with that text and the model continues it; the returned
        GenerationResult.text contains ONLY the continuation.
        """

    # --- convenience ------------------------------------------------------- #
    def complete(self, prompt: str, **kwargs) -> GenerationResult:
        """Raw text completion (base models / non-chat use)."""
        return self.chat([ChatMessage("user", prompt)], **kwargs)

    def close(self) -> None:  # pragma: no cover - backend specific
        pass
