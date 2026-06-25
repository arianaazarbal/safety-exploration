"""Paraphrasing of truncated responses (Section 3.1 / Appendix C.2).

Paraphrases the truncated assistant text with Claude-Sonnet to control for
stylistic biases from using Gemma-generated text, preserving meaning, tone, and
emotion level, and keeping the text ending at roughly the same point.
"""
from __future__ import annotations

from ..config import load_prompt

_PARAPHRASE_PROMPT = load_prompt("paraphrase.txt")


def paraphrase(paraphraser_model, text: str) -> str:
    prompt = _PARAPHRASE_PROMPT.replace("{text}", text)
    reply = paraphraser_model.chat(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024
    )
    return reply.strip()
