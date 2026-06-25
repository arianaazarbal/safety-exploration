"""Paraphrasing of truncated responses (Appendix C.2).

Paraphrasing controls for stylistic biases from Gemma-generated text so the
base/instruct comparison reflects emotional content, not surface style. Meaning,
tone and emotion level are preserved; the text is left ending at roughly the
same (possibly mid-sentence) point.
"""

from __future__ import annotations

import time

import config

# Appendix C.2 — verbatim paraphrase prompt.
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
    def __init__(self, model: str = config.JUDGE_MODEL):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(b.text for b in msg.content if b.type == "text").strip()
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return text
