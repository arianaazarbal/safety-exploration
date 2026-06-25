"""Anthropic backend for the Claude judge and Petri auditor/judge.

Used only for infrastructure roles, never as a target model:

* Frustration judge — ``claude-sonnet-4-20250514`` (Appendix B.2).
* Onset labelling / paraphrasing — ``claude-sonnet-4-20250514`` (Appendix C).
* Petri auditor — ``claude-sonnet-4-20250514``; Petri judge — ``claude-opus-4-20250514``
  (Appendix G).

These are the *exact* historical model IDs the paper pinned. We keep them verbatim rather
than upgrading to the current default model, because the point of a replication is to
reproduce the paper's measurements (the judge model materially affects scores). DESIGN.md
explains this choice and how to override it. These IDs use the classic Messages API
surface (no adaptive thinking), so a plain ``messages.create`` with ``temperature`` and
``max_tokens`` is correct here.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..utils import env, parallel_map, with_retries
from .base import ChatModel, Conversation

logger = logging.getLogger(__name__)


class AnthropicBackend(ChatModel):
    """Claude model accessed via the official Anthropic SDK."""

    supports_prefill = False

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        max_workers: int = 8,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        super().__init__(name)
        import anthropic

        self.model_id = model_id
        self.max_workers = max_workers
        self._client = anthropic.Anthropic(api_key=env(api_key_env, required=True))

    @staticmethod
    def _split_system(conversation: Conversation) -> tuple[Optional[str], Conversation]:
        """Extract a leading system message (the Anthropic API takes ``system`` separately)."""
        if conversation and conversation[0]["role"] == "system":
            return conversation[0]["content"], conversation[1:]
        return None, list(conversation)

    def _one(
        self, conversation: Conversation, *, temperature: float, max_new_tokens: int
    ) -> str:
        system, messages = self._split_system(conversation)

        def call() -> str:
            kwargs = dict(
                model=self.model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                messages=messages,  # type: ignore[arg-type]
            )
            if system is not None:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text")

        return with_retries(call, label=f"anthropic:{self.model_id}")

    def chat_batch(
        self,
        conversations: list[Conversation],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        tasks = [(ci, conv) for ci, conv in enumerate(conversations) for _ in range(n)]
        outputs = parallel_map(
            lambda t: self._one(
                t[1], temperature=temperature, max_new_tokens=max_new_tokens
            ),
            tasks,
            max_workers=self.max_workers,
            desc=f"anthropic:{self.name}",
        )
        grouped: list[list[str]] = [[] for _ in conversations]
        for (ci, _), out in zip(tasks, outputs):
            grouped[ci].append(out)
        return grouped
