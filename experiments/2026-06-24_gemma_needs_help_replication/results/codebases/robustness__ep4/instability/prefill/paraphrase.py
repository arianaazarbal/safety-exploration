"""Paraphrasing of truncated prefills (Appendix C.2).

To control for stylistic biases from Gemma-generated text, every truncation is
paraphrased by Claude Sonnet, preserving meaning/emotion level and ending at
roughly the same point.
"""
from __future__ import annotations

from ..config import ONSET_LABELLER
from ..models.registry import load_model

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase(text: str, model=None) -> str:
    model = model or load_model(ONSET_LABELLER)
    out = model.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_new_tokens=1024, n=1,
    )[0].text
    return out.strip()
