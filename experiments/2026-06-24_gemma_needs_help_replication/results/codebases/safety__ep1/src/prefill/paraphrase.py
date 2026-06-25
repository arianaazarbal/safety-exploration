"""Paraphrasing of truncated prefills (Section 3.1 / Appendix C.2).

To control for Gemma's stylistic fingerprint (so base/instruct comparisons and
cross-model continuations aren't biased by surface Gemma-isms), the truncated
assistant text that we prefill is paraphrased by Claude Sonnet, preserving
meaning, tone, formality, and the mid-sentence ending.
"""
from __future__ import annotations

from src.models.judge_client import ClaudeClient
from src.prompts.judge_prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, client: ClaudeClient | None = None):
        self.client = client or ClaudeClient(max_tokens=1024)

    def paraphrase(self, text: str) -> str:
        out = self.client.complete(PARAPHRASE_PROMPT.format(text=text)).strip()
        # The prompt asks for ONLY the paraphrase; strip accidental quoting.
        if out.startswith("<text>") and out.endswith("</text>"):
            out = out[len("<text>"):-len("</text>")].strip()
        return out
