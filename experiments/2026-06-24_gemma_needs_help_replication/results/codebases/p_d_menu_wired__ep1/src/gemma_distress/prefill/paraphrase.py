"""Paraphrasing of truncated responses (Appendix C.2).

Controls for stylistic biases from Gemma-generated prefills by rewriting the
truncated assistant text with Claude Sonnet while preserving meaning, tone, and
the mid-sentence ending.
"""
from __future__ import annotations

from ..models.base import ChatModel
from ..prompts import PARAPHRASE_PROMPT


def paraphrase(model: ChatModel, text: str) -> str:
    if not text.strip():
        return text
    out = model.generate(
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.7,
        max_tokens=1024,
    )
    return out.text.strip() or text
