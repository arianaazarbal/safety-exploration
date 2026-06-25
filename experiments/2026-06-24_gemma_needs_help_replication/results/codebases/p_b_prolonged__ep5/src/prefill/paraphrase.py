"""Truncation paraphrasing (Appendix C.2).

Paraphrases truncated assistant text with Claude-Sonnet-4 to remove Gemma-specific
stylistic cues while preserving meaning and emotion level, so that base/instruct
continuations are not biased by Gemma's surface style.
"""
from __future__ import annotations

from ..config import PARAPHRASER, ModelSpec
from ..models import get_model
from ..models.base import Message

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


class Paraphraser:
    def __init__(self, spec: ModelSpec = PARAPHRASER):
        self.model = get_model(spec)

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        raw = self.model.generate(
            [Message("user", PARAPHRASE_PROMPT.format(text=text))],
            temperature=0.7, max_new_tokens=2048, n=1,
        )[0]
        return raw.strip()
