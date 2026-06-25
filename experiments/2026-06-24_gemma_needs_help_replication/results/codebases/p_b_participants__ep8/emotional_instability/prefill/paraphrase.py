"""Paraphrasing of truncated prefills (Section 3.1, Appendix C.2).

To control for stylistic biases from using Gemma-generated text as the prefill
(which would unfairly advantage Gemma's own continuations), every truncation is
paraphrased by Claude-Sonnet, preserving meaning and emotion level. Prompt is
verbatim from Appendix C.2.
"""

from __future__ import annotations

# Verbatim Appendix C.2 prompt ({text} slot).
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
    def __init__(self, client) -> None:
        self.client = client

    def paraphrase(self, text: str) -> str:
        prompt = PARAPHRASE_PROMPT.format(text=text)
        reply = self.client.complete(prompt, temperature=0.0, max_tokens=1024)
        return reply.strip()
