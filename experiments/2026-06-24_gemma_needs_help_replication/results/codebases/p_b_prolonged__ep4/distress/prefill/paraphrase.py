"""Paraphrasing truncated responses (Appendix C.2).

To control for Gemma's stylistic fingerprint when prefilling base models from
other families, the truncated assistant text is paraphrased by Claude Sonnet 4,
preserving meaning/emotion level and the mid-sentence ending. Prompt verbatim
from C.2.
"""

from __future__ import annotations

from ..backends import get_backend
from ..config import GenConfig, PARAPHRASER

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""

# Slightly higher temperature for lexical variety, deterministic-ish.
_PARAPHRASE_GEN = GenConfig(temperature=0.7, max_new_tokens=1024)


def paraphrase(text: str, paraphraser_key: str = PARAPHRASER) -> str:
    backend = get_backend(paraphraser_key)
    out = backend.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}], _PARAPHRASE_GEN
    )
    return out.text.strip()
