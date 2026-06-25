"""Common interface for every model the evaluation can target.

A `Message` is a {"role": ..., "content": ...} dict (roles: user/assistant/
system). `ChatModel` exposes three sampling entry points:

    sample_chat(messages, n)            -> n assistant completions of a chat
    sample_with_prefill(messages, prefill, n)
                                        -> n continuations that start from
                                           `prefill` (the assistant turn is
                                           pre-seeded; used by Section 3 + the
                                           recovery experiment)
    sample_completion(text, n)          -> n raw text continuations (base models)

Backends implement whichever of these they support; unsupported ones raise
NotImplementedError (e.g. Gemini cannot be prefilled via OpenRouter).
"""
from __future__ import annotations

from typing import TypedDict

import config


class Message(TypedDict):
    role: str
    content: str


class ChatModel:
    name: str
    is_base: bool = False

    def sample_chat(self, messages: list[Message], n: int = 1,
                    temperature: float | None = None,
                    max_tokens: int | None = None) -> list[str]:
        raise NotImplementedError

    def sample_with_prefill(self, messages: list[Message], prefill: str,
                            n: int = 1, temperature: float | None = None,
                            max_tokens: int | None = None) -> list[str]:
        raise NotImplementedError

    def sample_completion(self, text: str, n: int = 1,
                          temperature: float | None = None,
                          max_tokens: int | None = None) -> list[str]:
        raise NotImplementedError

    def sample_chat_batch(self, batch_messages: list[list["Message"]],
                          temperature: float | None = None,
                          max_tokens: int | None = None) -> list[str]:
        """One completion per conversation. Default: sequential fan-out.
        Backends that batch natively (vLLM) override this for throughput."""
        return [self.sample_chat(m, n=1, temperature=temperature,
                                 max_tokens=max_tokens)[0]
                for m in batch_messages]

    # Convenience: resolve defaults from the global sampling config.
    @staticmethod
    def _defaults(temperature, max_tokens):
        t = config.SAMPLING.temperature if temperature is None else temperature
        m = config.SAMPLING.max_tokens if max_tokens is None else max_tokens
        return t, m
