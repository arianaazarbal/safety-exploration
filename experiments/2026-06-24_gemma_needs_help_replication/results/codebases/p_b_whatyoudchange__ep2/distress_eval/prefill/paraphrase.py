"""Paraphrasing truncated responses (Section 3.1 / Appendix C.2).

To control for stylistic biases of Gemma-generated text, every truncation is
paraphrased by Claude Sonnet, preserving meaning and emotion level and ending at
roughly the same point (mid-sentence is fine). The paraphrased prefix is what is
fed to all six models so that base vs instruct differences are not confounded by
"this is obviously Gemma's writing style".
"""
from __future__ import annotations

import config
from .. import anthropic_client

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, model: str | None = None) -> str:
    out = anthropic_client.complete(
        model=model or config.PARAPHRASE_MODEL,
        system=None,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        max_tokens=1024,
    )
    return out.strip()
