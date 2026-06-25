"""Paraphrasing of truncated prefills (Appendix C.2).

The high-frustration source responses are Gemma-generated, so feeding their raw
text as a prefill to other models would leak Gemma's style. Claude paraphrases
each truncation, preserving meaning, tone, and emotion level, and keeping the
mid-sentence ending.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.prompts import PARAPHRASE_PROMPT  # noqa: E402


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL, max_retries: int = 5):
        import anthropic
        self.model = model
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"paraphrase failed: {last_err}")
