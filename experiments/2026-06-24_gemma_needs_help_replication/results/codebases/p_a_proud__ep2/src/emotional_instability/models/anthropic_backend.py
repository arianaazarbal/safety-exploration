"""Anthropic backend for the judge / auditor / paraphraser / onset-labeller roles.

Uses claude-sonnet-4-20250514 (judge, auditor, onset, paraphrase) and
claude-opus-4-20250514 (Petri judge), per Appendices B.2, C and G.
"""
from __future__ import annotations

import os
from functools import cached_property

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ModelSpec
from ..utils import Message
from .base import GenerationError, ModelBackend


class AnthropicBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, *, api_key: str | None = None):
        super().__init__(spec)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @cached_property
    def _client(self):
        import anthropic
        if not self._api_key:
            raise GenerationError("ANTHROPIC_API_KEY is not set.")
        return anthropic.Anthropic(api_key=self._api_key)

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """Anthropic takes the system prompt as a separate top-level argument."""
        if messages and messages[0]["role"] == "system":
            return messages[0]["content"], messages[1:]
        return None, list(messages)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def chat(self, messages: list[Message], *, temperature=None, max_tokens=None) -> str:
        system, convo = self._split_system(messages)
        kwargs: dict = {
            "model": self.spec.model_id,
            "messages": convo,
            "temperature": self._temperature(temperature),
            "max_tokens": self._max_tokens(max_tokens),
        }
        if system is not None:
            kwargs["system"] = system
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise GenerationError(f"Anthropic call failed for {self.spec.name}: {e}") from e
        # Concatenate text blocks (ignore any tool/thinking blocks).
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text:
            raise GenerationError(f"Anthropic returned no text for {self.spec.name}.")
        return text.strip()
