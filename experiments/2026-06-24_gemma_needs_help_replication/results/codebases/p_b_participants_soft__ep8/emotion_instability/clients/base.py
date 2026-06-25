"""Common chat-client interface.

All backends speak the same small message protocol so the rollout / judge /
auditor code is backend-agnostic.  The one capability that differs is
*prefill* (forcing an assistant prefix that the model continues) -- only the
local HF backend supports it, which is exactly why the Section 3 experiment is
Gemma-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    thinking: bool = False
    stop: Sequence[str] | None = None


class ChatClient:
    """Abstract chat client."""

    #: whether ``generate(..., prefill=...)`` is honoured at the token level
    supports_prefill: bool = False

    def __init__(self, model_id: str, name: str | None = None):
        self.model_id = model_id
        self.name = name or model_id

    def generate(self, messages: list[Message], cfg: GenConfig,
                 prefill: str | None = None) -> str:
        """Return the assistant completion for `messages`.

        If `prefill` is given and the backend supports it, the assistant turn is
        forced to *begin* with `prefill` and the model continues from there; the
        returned string is the continuation **including** the prefill, unless a
        backend documents otherwise.
        """
        raise NotImplementedError

    # token utilities (used by the prefill experiment); default to a whitespace
    # approximation, overridden by HF backend with the real tokenizer.
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        return " ".join(text.split()[:n_tokens])
