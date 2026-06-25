"""Paraphrase prompt for truncated prefills (Appendix C.2, verbatim)."""

from __future__ import annotations

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def build_paraphrase_input(text: str) -> str:
    return PARAPHRASE_PROMPT.format(text=text)
