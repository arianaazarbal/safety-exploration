"""Abstract model-client interface.

Every model the paper elicits distress from (Gemma local, Gemini API) is exposed
through the same small surface so the eval runner is backend-agnostic:

    chat(messages)            -> assistant string   (instruct / chat models)
    continue_from(messages,   -> assistant string   (prefilling: the model must
                  prefill)                            *continue* `prefill` rather
                                                      than start a fresh turn)

`messages` follows the OpenAI-style role/content convention:
    [{"role": "system"|"user"|"assistant", "content": str}, ...]
"""

from __future__ import annotations

import abc
from typing import Sequence

Message = dict[str, str]


class ModelClient(abc.ABC):
    """Common interface for distress-elicitation targets."""

    name: str

    @abc.abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Return the assistant's reply to a chat-formatted conversation."""

    def continue_from(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Continue an assistant turn that already begins with `prefill`.

        Returns ONLY the continuation (the prefill is stripped), matching the
        paper's "score the generated continuation excluding prefill" (§3.1).

        Default implementation raises; only backends that can genuinely prefill
        (local Gemma) override it. API models generally cannot, which is exactly
        why the §3 base-vs-instruct study is restricted to local models.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling"
        )

    def close(self) -> None:  # pragma: no cover - resource cleanup hook
        """Free GPU memory / connections. No-op by default."""
