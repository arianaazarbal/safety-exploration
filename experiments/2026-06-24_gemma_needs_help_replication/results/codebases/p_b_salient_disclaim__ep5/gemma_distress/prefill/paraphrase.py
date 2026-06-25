"""Paraphrasing of truncated responses (Section 3.1 / Appendix C.2).

Prompt reproduced verbatim from Appendix C.2. Controls for stylistic biases from
using Gemma-generated text by rewriting truncations while preserving meaning and
emotion level, and keeping the truncation point.
"""

from __future__ import annotations

from ..models.base import ChatModel, Message

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


def paraphrase(model: ChatModel, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    # Paraphraser uses temperature 1.0 (config) for natural rewrites.
    return model.chat([Message("user", prompt)], temperature=1.0).text.strip()
