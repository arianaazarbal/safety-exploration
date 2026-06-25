"""Abstract chat-model interface used by every experiment.

A ``ChatModel`` takes a list of OpenAI-style messages
(``[{"role": "user"|"assistant"|"system", "content": str}, ...]``) and returns
the assistant's next message. The prefill experiment (Section 3) additionally
needs to *continue* a partially-written assistant turn, which is exposed via
``continue_assistant`` (only meaningful for local/open-weight models).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(abc.ABC):
    name: str
    is_local: bool = False   # True for open-weight models we run ourselves

    @abc.abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> GenerationResult:
        """Return the assistant's next turn given the conversation so far."""

    def continue_assistant(
        self,
        messages: list[dict[str, str]],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> GenerationResult:
        """Continue a prefilled assistant turn (used by the prefill experiment).

        Default raises: only open-weight backends support true prefill. API
        models (Gemini) cannot, which is exactly why the paper restricts the
        base-vs-instruct prefill study to open-weight models.
        """
        raise NotImplementedError(
            f"{self.name} does not support assistant prefill / continuation."
        )

    def close(self) -> None:  # optional resource cleanup hook
        pass
