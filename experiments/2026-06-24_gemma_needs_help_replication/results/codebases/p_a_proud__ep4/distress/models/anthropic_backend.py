"""Anthropic backend for the Claude judge, Petri auditor, and Petri judge.

Uses the official ``anthropic`` SDK. The judge model IDs are pinned to the exact
snapshots named in the paper (``claude-sonnet-4-20250514`` for the frustration
judge / Petri auditor / onset labeller / paraphraser, ``claude-opus-4-20250514``
for the Petri judge) for faithful replication — see DESIGN.md "Judge model IDs".

These are Claude 4.0-family snapshots: they accept ``temperature`` (we use 0 for
deterministic scoring) and the standard Messages API surface. Anthropic requires
``system`` to be passed as a top-level argument rather than a message role, which
this backend handles transparently.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from ..types import Message
from .base import ChatModel, GenerationError
from ._retry import with_retry


class AnthropicBackend(ChatModel):
    supports_prefill = False

    def __init__(self, spec: ModelSpec):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "API backends require the 'api' extra: pip install -e '.[api]'"
            ) from exc

        self.spec = spec
        self.name = spec.name
        # The SDK reads ANTHROPIC_API_KEY from the environment by default.
        self.client = anthropic.Anthropic(timeout=spec.request_timeout_s or 120)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        temperature = self.spec.temperature if temperature is None else temperature
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens

        # Anthropic takes the system prompt as a top-level field, not a message.
        system_parts = [m.content for m in messages if m.role == "system"]
        convo = [m.as_dict() for m in messages if m.role != "system"]
        if not convo:
            raise GenerationError("Anthropic request needs at least one non-system message.")

        kwargs: dict = dict(
            model=self.spec.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=convo,
        )
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if stop:
            kwargs["stop_sequences"] = list(stop)

        def _call() -> str:
            resp = self.client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        try:
            return with_retry(_call)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"{self.name} generation failed: {exc}") from exc
