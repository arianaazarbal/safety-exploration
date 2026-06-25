"""Paraphrasing of truncated responses (Section 3.1, Appendix C.2).

"To mitigate stylistic biases from using Gemma-generated responses, we paraphrase
all truncations using Claude Sonnet, in a manner that preserves the same meaning
and level of emotions."
"""
from __future__ import annotations

import time

import config

# Verbatim from Appendix C.2.
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
    def __init__(self, model: str | None = None, max_retries: int = 4):
        import anthropic

        self.model = model or config.JUDGE_MODEL
        self.max_retries = max_retries
        self._client = anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"paraphrase failed: {last_exc}")
