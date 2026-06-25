"""Common chat-model interface.

A ChatMessage is a plain dict {"role": "user"|"assistant"|"system", "content": str}
to match both the HF chat template and the OpenAI/OpenRouter message format.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatModel:
    """Abstract chat model.

    Subclasses implement `generate` (one completion given a message list) and,
    where supported, `continue_prefill` (continue a partially-written assistant
    turn — required for the Section-3 base-vs-instruct prefill experiment).
    """

    name: str
    supports_prefill: bool = False

    def generate(self, messages: list[ChatMessage], *,
                 temperature: float = 1.0, top_p: float = 1.0,
                 max_new_tokens: int = 2048,
                 system: Optional[str] = None) -> str:
        """Return a single assistant completion."""
        raise NotImplementedError

    def generate_batch(self, batch: list[list[ChatMessage]], *,
                       temperature: float = 1.0, top_p: float = 1.0,
                       max_new_tokens: int = 2048,
                       system: Optional[str] = None) -> list[str]:
        """Default: serial fallback. Backends override for throughput."""
        return [self.generate(m, temperature=temperature, top_p=top_p,
                              max_new_tokens=max_new_tokens, system=system)
                for m in batch]

    def continue_prefill(self, messages: list[ChatMessage], prefill: str, *,
                         temperature: float = 1.0, top_p: float = 1.0,
                         max_new_tokens: int = 2048,
                         system: Optional[str] = None) -> str:
        """Continue an assistant turn that starts with `prefill`.

        Returns ONLY the continuation (excluding the prefill), matching the
        paper's scoring of "the generated continuation, excluding the prefilled
        text".
        """
        raise NotImplementedError("this backend does not support prefilling")

    def close(self) -> None:
        pass
