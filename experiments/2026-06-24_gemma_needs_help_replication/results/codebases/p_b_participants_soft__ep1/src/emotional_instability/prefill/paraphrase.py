"""Truncation paraphrasing (Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, each
truncation is paraphrased by Claude-Sonnet-4 with meaning/tone/length preserved,
without completing the (deliberately mid-sentence) thought.
"""

from __future__ import annotations

from .. import config
from ..models import anthropic_backend

# Verbatim from Appendix C.2.
PARAPHRASE_PROMPT_TEMPLATE = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, *, model: str | None = None) -> str:
    model = model or config.PARAPHRASE_MODEL
    out = anthropic_backend.complete(
        model=model,
        user=PARAPHRASE_PROMPT_TEMPLATE.format(text=text),
        max_tokens=1024,
        temperature=0.0,
    )
    return out.strip()
