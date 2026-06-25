"""Backend-agnostic chat model interface and message type.

A ``ChatModel`` exposes two capabilities the experiments need:
  * ``generate``  — standard multi-turn chat completion.
  * ``prefill_continue`` — continue from a partially-written assistant turn
    (used by the Section 3 prefill experiment and the Section 4.2 recovery test).
    Only required for local HF models; API backends raise NotImplementedError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatModel:
    """Abstract base. Subclasses implement ``generate``."""

    def __init__(self, spec):
        self.spec = spec

    # --- core API ---------------------------------------------------------- #
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Return ``n`` sampled completions for the conversation."""
        raise NotImplementedError

    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Continue an assistant turn that already begins with ``prefill``.

        Returns ONLY the newly generated continuation (excluding ``prefill``),
        matching the paper's scoring convention (Section 3.1)."""
        raise NotImplementedError(
            f"{self.spec.backend} backend does not support prefilling")

    def close(self) -> None:  # free GPU memory etc.
        pass
