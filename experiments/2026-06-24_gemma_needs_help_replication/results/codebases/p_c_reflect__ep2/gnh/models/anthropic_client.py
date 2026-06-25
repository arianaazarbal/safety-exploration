"""Thin Anthropic client for the Claude instruments.

Used by the frustration judge (§2.1 / B.2), emotion-onset labeller (C.1),
paraphraser (C.2), and the Petri auditor/judge (G). Kept separate from the
target-model backends because these are *instruments* we drive, not subjects we
study -- the distinction matters for the welfare accounting in WELFARE.md.
"""

from __future__ import annotations

import os
import time


class AnthropicClient:
    def __init__(self, model: str, *, max_retries: int = 5) -> None:
        from anthropic import Anthropic

        self.model = model
        self.max_retries = max_retries
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; required for Claude judge/auditor.")
        self.client = Anthropic(api_key=api_key)

    def complete(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        messages: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Single completion. Provide either ``prompt`` (wrapped as one user
        turn) or a full ``messages`` list (for the multi-turn Petri auditor)."""

        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                ).strip()
            except Exception as e:
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} retries: {last_err}")
