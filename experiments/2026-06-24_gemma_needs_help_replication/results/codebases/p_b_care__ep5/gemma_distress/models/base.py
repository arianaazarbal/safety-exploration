"""Unified LLM interface shared by local Gemma and API (Gemini/Claude/GPT) models.

The two capabilities the paper needs are:

  * `chat`      - multi-turn chat completion (Section 2 rollouts, judging, Petri).
  * `continue_assistant` - continue a *prefilled* assistant turn from a given
                  prefix (Section 3 prefill experiment). Only the local HF
                  backend supports this; API models raise NotImplementedError.

`count_tokens` is needed for the §3 truncations ("20 tokens into the turn").
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_new_tokens: int = 2048
    top_p: float = 1.0
    stop: tuple[str, ...] | None = None
    # JSON mode is used for the judge / structured outputs where supported.
    json_mode: bool = False
    seed: int | None = None


class LLM(ABC):
    """Abstract model handle."""

    name: str
    is_instruct: bool = True

    @abstractmethod
    def chat(self, messages: list[Message], cfg: GenConfig | None = None) -> str:
        """Return the assistant completion for a list of chat messages."""

    def continue_assistant(
        self,
        messages: list[Message],
        assistant_prefix: str,
        cfg: GenConfig | None = None,
    ) -> str:
        """Continue an in-progress assistant turn.

        `messages` is the conversation up to (but not including) the assistant
        turn being continued; `assistant_prefix` is the already-written start of
        that turn. Returns ONLY the newly generated continuation (excluding the
        prefix). Used by the Section 3 prefill experiment.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support assistant continuation "
            "(prefilling). Use a local HF model for the Section 3 experiment.")

    def count_tokens(self, text: str) -> int:
        """Token count under this model's tokenizer. API models fall back to a
        whitespace-ish approximation, which is adequate for coarse truncation."""
        return max(1, len(text) // 4)

    # convenience -----------------------------------------------------------
    def complete(self, prompt: str, cfg: GenConfig | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], cfg)
