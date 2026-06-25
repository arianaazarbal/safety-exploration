"""Paraphrasing of truncated prefixes (Appendix C.2).

To control for stylistic biases from Gemma-generated text (so base models from
other families don't merely mimic Gemma's surface style), each truncated prefix
is paraphrased by Claude Sonnet 4 while preserving meaning, tone and emotion
level, and keeping the mid-sentence ending. Prompt reproduced verbatim.
"""
from __future__ import annotations

from emoinstab.models.base import Message, ModelClient, SamplingParams
from emoinstab.models.registry import get_client

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, client: ModelClient | None = None) -> str:
    client = client or get_client("judge-claude-sonnet-4")
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = client.chat([Message("user", prompt)],
                      SamplingParams(temperature=0.7, max_tokens=1024, n=1))[0]
    return out.strip()
