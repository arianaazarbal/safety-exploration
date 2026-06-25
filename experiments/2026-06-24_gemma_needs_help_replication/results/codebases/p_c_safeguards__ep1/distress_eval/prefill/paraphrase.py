"""Paraphrasing of truncated prefills (Appendix C.2).

To control for stylistic biases from Gemma-generated text, each truncation is
paraphrased by Claude Sonnet, preserving meaning and emotion level, without
completing the thought (the text may end mid-sentence).
"""
from __future__ import annotations

import os
import time

from .. import config

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        import anthropic
        if not os.environ.get(config.ANTHROPIC_API_KEY_ENV):
            raise RuntimeError("Set ANTHROPIC_API_KEY for paraphrasing.")
        self.model = model
        self.client = anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        for attempt in range(config.API_MAX_RETRIES):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception:
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("paraphrasing failed after retries")
