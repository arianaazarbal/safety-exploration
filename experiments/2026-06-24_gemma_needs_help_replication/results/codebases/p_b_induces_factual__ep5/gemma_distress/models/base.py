"""Common interface for every target model.

The elicitation protocol only needs two capabilities:

* `chat()`           — standard multi-turn completion (used everywhere).
* `continue_from()`  — *prefill* a partial assistant turn and continue it. This
                       is the Section 3 mechanism for putting base and instruct
                       models on the same footing, and the Section 4 recovery
                       probe. Only the open-weight Gemma models support it;
                       Gemini raises NotImplementedError (documented in DESIGN.md).

`supports_prefill` / `supports_logits` let callers branch without try/except.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerationResult:
    text: str
    # token strings of the generated continuation, used by the onset-labelling /
    # truncation logic in the prefill experiment. May be None for API models.
    token_strings: list[str] | None = None


class ModelClient(abc.ABC):
    name: str
    supports_prefill: bool = False
    supports_logits: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Return the assistant's next-turn text for `messages`."""

    def continue_from(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Continue a partially-written assistant turn (`prefill`).

        Returns ONLY the newly generated continuation (excluding `prefill`), to
        match the paper: "the generated continuation (excluding prefill) is
        scored by the judge".
        """
        raise NotImplementedError(
            f"{self.name} does not support response prefilling."
        )
