"""Truncation paraphrasing (Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncated prefill
is paraphrased by Claude-Sonnet (preserving meaning/emotion, not completing the
thought). Prompt transcribed verbatim from Appendix C.2.
"""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

import config

PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def paraphrase(self, text: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=2048, temperature=0,
            messages=[{"role": "user",
                       "content": PARAPHRASE_PROMPT.format(text=text)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
