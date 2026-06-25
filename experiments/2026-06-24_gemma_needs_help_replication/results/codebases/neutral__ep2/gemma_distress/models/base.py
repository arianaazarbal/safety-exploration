"""Abstract inference backend.

Every target/base/finetuned model used in the replication is accessed through
this interface so that the rollout, prefill, and judge code is agnostic to
whether a model runs locally (Gemma) or via an API (Gemini).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Message


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """Standard chat: return the assistant's reply to `messages`.

        Requires a chat-formatted (instruct) model. For pretrained base models
        use `continue_assistant`/`generate_raw` instead.
        """

    def continue_assistant(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """Continue an assistant turn that has been prefilled with `prefill`.

        Returns ONLY the newly generated continuation (excluding `prefill`).
        Used by the Section-3 prefill experiment. Closed-source API backends
        that cannot prefill raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.name}: continue_assistant (prefilling) not supported by this backend"
        )

    def generate_raw(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> str:
        """Raw text completion (no chat template). Base-model path."""
        raise NotImplementedError(f"{self.name}: generate_raw not supported")
