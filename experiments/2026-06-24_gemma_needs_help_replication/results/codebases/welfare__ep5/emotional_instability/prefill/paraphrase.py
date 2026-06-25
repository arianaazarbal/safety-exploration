"""Paraphrase truncated prefills (Section 3.1, Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude Sonnet before being used as a prefill for *all* models
(base and instruct, both families). Meaning, tone, and the mid-sentence ending
are preserved.
"""

from __future__ import annotations

from typing import Optional

from .. import config
from ..models.base import ChatModel
from ..models.registry import load_model

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
Respond with ONLY the paraphrased text, nothing else.
"""


def paraphrase_truncation(text: str, model: Optional[ChatModel] = None) -> str:
    if not text.strip():
        return text
    model = model or load_model(config.PARAPHRASE_MODEL)
    reply = model.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_new_tokens=1024, n=1,
    )[0]
    return reply.strip()
