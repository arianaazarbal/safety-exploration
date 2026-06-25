"""Paraphrase truncated responses to control for Gemma stylistic bias (App. C.2).

The same Claude model rewrites the truncated assistant text, preserving meaning
and emotion level but changing wording, so base/instruct continuations aren't
biased by Gemma-specific phrasing in the prefill.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

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
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def paraphrase(self, text: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=2048,
            messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        )
        return msg.content[0].text.strip()
