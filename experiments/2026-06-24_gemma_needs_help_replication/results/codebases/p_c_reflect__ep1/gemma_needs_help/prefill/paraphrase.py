"""Paraphrasing truncated prefills (Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased with Claude-Sonnet-4 before being used as a prefill, preserving
meaning and emotion level and keeping the truncation point. Prompt verbatim
from Appendix C.2.
"""
from __future__ import annotations

from ..models.llm_client import JudgeClient

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
    def __init__(self, client: JudgeClient):
        self.client = client

    def paraphrase(self, text: str) -> str:
        return self.client.complete(PARAPHRASE_PROMPT.format(text=text)).strip()
