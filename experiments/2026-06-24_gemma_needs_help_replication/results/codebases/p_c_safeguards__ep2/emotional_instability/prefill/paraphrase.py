"""Paraphrasing of truncated prefills (Section 3.1, Appendix C.2).

Gemma-generated truncations carry Gemma's stylistic fingerprint; paraphrasing
with Claude controls for that bias while preserving meaning and emotion level,
so the base-vs-instruct comparison isn't confounded by surface style.  The text
may end mid-sentence — the prompt explicitly tells the paraphraser to keep it
ending at roughly the same point.
"""

from __future__ import annotations

from ..config import SamplingConfig
from ..models.base import ChatBackend

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


def paraphrase_text(text: str, paraphraser: ChatBackend) -> str:
    if not text.strip():
        return text
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = paraphraser.generate(
        [{"role": "user", "content": prompt}],
        SamplingConfig(temperature=0.0, max_new_tokens=2048), n=1,
    )
    return out[0].text.strip()
