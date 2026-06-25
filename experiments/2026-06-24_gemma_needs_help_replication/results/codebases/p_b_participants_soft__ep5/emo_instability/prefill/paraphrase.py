"""Truncation paraphrasing (Section 3.1 / Appendix C.2).

To mitigate stylistic biases from Gemma-generated text, every prefill is
paraphrased by Claude Sonnet while preserving meaning, tone, and the
mid-sentence truncation point.
"""
from __future__ import annotations

from ..models import infrastructure_client
from ..models.base import ChatClient
from ..prompts.judge_prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or infrastructure_client("paraphraser")

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        out = self.client.generate(
            [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=1024
        )
        return out.strip()
