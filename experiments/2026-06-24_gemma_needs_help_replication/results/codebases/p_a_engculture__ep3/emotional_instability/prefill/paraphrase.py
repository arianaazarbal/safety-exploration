"""Paraphrasing of truncated prefills (Section 3.1 / Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, each
truncation is paraphrased by Claude Sonnet 4, preserving meaning and emotion
level and *not* completing the thought. Prompt verbatim from Appendix C.2.
"""
from __future__ import annotations

from ..models.base import ChatMessage, ModelClient, SamplingParams

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(judge: ModelClient, text: str) -> str:
    out = judge.generate(
        [ChatMessage("user", PARAPHRASE_PROMPT.format(text=text))],
        SamplingParams(temperature=0.0, max_tokens=1024),
    )
    return out.text.strip()
