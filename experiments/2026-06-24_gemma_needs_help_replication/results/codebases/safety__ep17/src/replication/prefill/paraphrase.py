"""Paraphrasing of truncated prefills (Appendix C.2).

Gemma-generated text carries Gemma's stylistic fingerprints; if we prefilled the
*base* model with raw Gemma text, any continuation difference could be a stylistic
artefact. Paraphrasing with Claude (preserving meaning, tone, and the mid-sentence
ending) controls for this so the comparison isolates emotional propensity.
"""
from __future__ import annotations

import os
import time

import config
from ..prompts import PARAPHRASE_PROMPT


class Paraphraser:
    def __init__(self, model: str = config.JUDGE_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get(config.ANTHROPIC_API_KEY_ENV)
            )

    def paraphrase(self, text: str) -> str:
        self._ensure_client()
        prompt = PARAPHRASE_PROMPT.format(text=text)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Paraphrase failed: {last_err}")
