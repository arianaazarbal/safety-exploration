"""Paraphrase truncated assistant text to remove Gemma stylistic fingerprints
(Section 3.1 / Appendix C.2). Verbatim prompt, Claude Sonnet 4."""

from __future__ import annotations

import os
import time

from ..prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, model="claude-sonnet-4-20250514", max_retries=5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=1024, temperature=0.7,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception:  # noqa: BLE001
                time.sleep(2 ** attempt)
        return text  # fall back to the original on persistent failure
