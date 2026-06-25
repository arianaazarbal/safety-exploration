"""Paraphrasing of truncated prefills (paper §3.1, Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, every
truncation is paraphrased by Claude Sonnet, preserving meaning and emotion level and
keeping the text ending at roughly the same point (mid-sentence is fine).
"""
from __future__ import annotations

from ..config import load_prompt
from ..models.base import ChatMessage, ModelClient

PARAPHRASE_PROMPT = load_prompt("paraphrase")


def paraphrase(model: ModelClient, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.replace("{text}", text)
    reply = model.chat([ChatMessage("user", prompt)], temperature=0.7, max_new_tokens=1024)
    return reply.strip()
