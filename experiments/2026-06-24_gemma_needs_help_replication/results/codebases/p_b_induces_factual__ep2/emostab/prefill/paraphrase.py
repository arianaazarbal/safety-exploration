"""Paraphrasing of truncated assistant text via Claude Sonnet (Appendix C.2).

Truncations are paraphrased to control for Gemma's stylistic fingerprint so that
base/instruct continuations are not biased by the surface form of the prefill.
"""
from __future__ import annotations

from ..utils.concurrency import with_retries
from .prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, provider="anthropic", model="claude-sonnet-4-20250514"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = PARAPHRASE_PROMPT.format(text=text)

        @with_retries
        def _call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

        return "".join(b.text for b in _call().content if b.type == "text").strip()
