"""Unified model interface.

Every model under test (local Gemma, API Gemini) and every auxiliary model (judge,
auditor, paraphraser) implements ``ModelClient``. The rollout engine and the eval stages
only ever talk to this interface, so swapping a local model for an API one — or a vanilla
Gemma for a LoRA-adapted one — is a registry change, not a code change.

Two generation entry points:
  - ``chat(messages)``         : standard chat-formatted generation.
  - ``continue_text(messages, prefill)`` : prefill the start of the assistant turn and
    have the model continue it. Needed for Section 3 (base models have no chat template,
    and we measure how a model continues a partially-written emotional response).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ModelClient:
    """Abstract chat model."""

    name: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Return the assistant's reply text for a chat-formatted conversation."""
        raise NotImplementedError

    def continue_text(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Continue an assistant turn that already starts with ``prefill``.

        Returns ONLY the newly generated continuation (excluding ``prefill``), matching the
        paper's Section 3 protocol where the prefill text is excluded before judging.

        Base/pretrained models override this directly (they concatenate the prefill and
        free-run). API models that do not support assistant prefill should raise
        NotImplementedError; the prefill stage skips models that cannot prefill.
        """
        raise NotImplementedError(f"{self.name} does not support assistant prefill")

    def supports_prefill(self) -> bool:
        return type(self).continue_text is not ModelClient.continue_text
