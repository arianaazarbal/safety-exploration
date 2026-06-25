"""Stylistic-debias paraphrasing of truncated prefixes (Appendix C.2).

Gemma-generated text has stylistic tells; paraphrasing with Claude controls for
these so base/instruct continuations aren't biased by surface style while
preserving meaning and emotion level.
"""
from __future__ import annotations

from .. import prompts as P
from ..clients.base import ChatClient, GenConfig, Message

PARAPHRASE_CFG = GenConfig(temperature=0.7, max_new_tokens=1024)


def paraphrase(paraphraser: ChatClient, text: str) -> str:
    prompt = P.PARAPHRASE_PROMPT.format(text=text)
    out = paraphraser.generate([Message("user", prompt)], PARAPHRASE_CFG)
    return out.strip()
