"""Anthropic API client used by the judge, paraphraser, onset labeller, and
Petri auditor/judge.

Kept separate from the target-model backends because these models are graders /
auditors, not subjects of the evaluation. Model IDs are pinned by the callers
(e.g. ``claude-sonnet-4-20250514`` for the judge) for reproducibility.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import Message


class AnthropicClient:
    def __init__(self, model_id: str):
        # Accept the registry's "anthropic/<id>" slug or a bare id.
        self.model_id = model_id.split("/", 1)[-1]
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self._ensure_client()
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=2, min=2, max=60))
        def _call():
            kwargs: dict = dict(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(block.text for block in resp.content if block.type == "text")

        return _call()
