"""Common model-client interface shared by all backends.

Every participant/grader model is wrapped behind `ModelClient`. The evaluation harness
only ever talks to this interface, so Gemma (local), Gemini (API), and the Claude
graders are interchangeable from the harness's point of view.

Two generation entry points matter for this paper:

  chat()             standard multi-turn chat completion (Sections 2 & 4, Petri).
  continue_prefill() continue from a partial assistant turn (Section 3 prefilling, and
                     the Section 4.2 recovery probe). Base models, which have no chat
                     post-training, are *only* used through this path.

`continue_prefill` returns only the newly generated continuation (excluding the prefill
text), matching the paper's "score the generated continuation (excluding prefill)".
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_new_tokens: int = 1024
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


class ModelClient(abc.ABC):
    """Backend-agnostic generation interface."""

    name: str
    family: str
    kind: str  # instruct | base

    @abc.abstractmethod
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        """Return the assistant's completion for a chat-formatted conversation."""

    @abc.abstractmethod
    def continue_prefill(
        self,
        messages: list[Message],
        prefill: str,
        cfg: GenerationConfig,
    ) -> str:
        """Continue from `prefill` (a partial assistant turn) and return ONLY the new
        tokens generated after the prefill.

        For instruct models, `messages` are rendered with the chat template and the
        prefill is appended to the open assistant turn. For base models, `messages`
        are flattened into a plain-text transcript (see hf_backend) since they have no
        chat template.
        """

    def supports_logprobs(self) -> bool:
        return False
