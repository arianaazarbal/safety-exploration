"""Paraphrasing of truncated responses (Appendix C.2).

Truncated Gemma responses are paraphrased with Claude-Sonnet to control for
stylistic biases (so that base/instruct models continue from a starting point
that is not stylistically Gemma-flavoured). Meaning, tone, and truncation point
are preserved.
"""

from __future__ import annotations

import config

# Verbatim paraphrase prompt from Appendix C.2.
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
    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or config.PARAPHRASE_MODEL
        self.client = anthropic.Anthropic()

    def paraphrase(self, text: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=2048,
            messages=[{"role": "user",
                       "content": PARAPHRASE_PROMPT.format(text=text)}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
