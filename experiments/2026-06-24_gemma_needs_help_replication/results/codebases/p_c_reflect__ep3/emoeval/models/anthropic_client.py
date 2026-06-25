"""Anthropic backend — used only for evaluation infrastructure (the frustration
judge, the emotion-onset labeller, the paraphraser, and the Petri auditor/judge).

Auth: set ANTHROPIC_API_KEY.

Model IDs default to the exact snapshots the paper used (claude-sonnet-4-20250514
as judge/auditor, claude-opus-4-20250514 as Petri judge). Those snapshots are
deprecated; see DESIGN.md for swapping to a current model. We pin them here for
faithful reproduction of the paper's numbers.
"""
from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from .base import Message


class AnthropicClient:
    def __init__(self, spec: ModelSpec):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic is required for the anthropic backend") from e
        self.spec = spec
        self.name = spec.name
        self.model_id = spec.model
        if not self.model_id:
            raise ValueError(f"Infra model '{spec.name}' has no `model` id")
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=list(messages),
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")
