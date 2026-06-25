"""Abstract model-client interface shared by local and API backends.

A `Message` is a plain dict {"role": "user"|"assistant"|"system", "content": str}.
The two capabilities the experiments need are:

  * generate(messages)              -- a normal assistant turn
  * generate(messages, prefill=...) -- continue an assistant turn that already
                                        starts with `prefill` text (used by the
                                        Section 3 prefill study); the returned
                                        text EXCLUDES the prefill.

Only the local HF backend supports prefill and hidden-state access; the API
backend raises NotImplementedError for those (Gemini is closed-source, matching
the paper's stated limitation that its base models cannot be studied).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

Message = dict  # {"role": str, "content": str}


class ModelClient(ABC):
    """Common interface for generating model responses."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        prefill: str | None = None,
    ) -> str:
        """Return the assistant's continuation text (excluding any prefill)."""
        raise NotImplementedError

    def generate_batch(
        self,
        batch_messages: list[list[Message]],
        *,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        """Default: sequential generation. HF backend overrides with true batching."""
        if prefills is None:
            prefills = [None] * len(batch_messages)
        return [
            self.generate(m, max_new_tokens=max_new_tokens,
                          temperature=temperature, prefill=p)
            for m, p in zip(batch_messages, prefills)
        ]

    @property
    def supports_prefill(self) -> bool:
        return False

    @property
    def supports_internals(self) -> bool:
        return False

    def close(self) -> None:  # release GPU memory etc.
        pass
