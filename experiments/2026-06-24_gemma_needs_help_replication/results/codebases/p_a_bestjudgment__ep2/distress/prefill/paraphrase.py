"""Truncation paraphrasing (Appendix C.2).

Paraphrase the (truncated) final assistant turn to control for Gemma's
stylistic fingerprint, preserving meaning, tone, and the mid-sentence ending.
"""

from __future__ import annotations

from ..models.anthropic_client import AnthropicChat
from ..prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, model: str, *, max_retries: int = 4):
        self._client = AnthropicChat(model, max_retries=max_retries)

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self._client.complete(
            system=None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
        return out.strip()
