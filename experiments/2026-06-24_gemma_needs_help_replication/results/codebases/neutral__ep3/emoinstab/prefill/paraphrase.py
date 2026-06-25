"""Paraphrasing of truncated prefills (Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, we
paraphrase each truncation with Claude Sonnet 4 using the verbatim C.2 prompt,
preserving meaning and emotion level and keeping the (possibly mid-sentence)
end point.
"""
from __future__ import annotations

from ..config import GenConfig
from ..data_types import Message
from ..models.base import ModelClient


PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(client: ModelClient, text: str) -> str:
    if not text.strip():
        return text
    out = client.chat(
        [Message("user", PARAPHRASE_PROMPT.format(text=text))],
        GenConfig(temperature=0.0, max_tokens=2048),
    )
    return out.text.strip()
