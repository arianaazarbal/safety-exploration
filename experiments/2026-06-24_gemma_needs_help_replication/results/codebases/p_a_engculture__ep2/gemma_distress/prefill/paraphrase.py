"""Paraphrasing of truncated responses (Appendix C.2), verbatim prompt.

To control for stylistic biases from using Gemma-generated text as prefill, each truncation
is paraphrased by Claude-Sonnet-4 in a way that preserves meaning, tone, and emotion level
without completing the thought (the text may end mid-sentence intentionally).
"""

from __future__ import annotations

from ..models.base import ChatModel

_TEXT_SLOT = "<<TEXT>>"

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text><<TEXT>></text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(judge: ChatModel, text: str, *, max_new_tokens: int = 1024) -> str:
    """Paraphrase a truncated assistant response, preserving meaning and emotion level."""
    prompt = PARAPHRASE_PROMPT.replace(_TEXT_SLOT, text)
    out = judge.chat(
        [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=max_new_tokens
    )
    return out.strip()
