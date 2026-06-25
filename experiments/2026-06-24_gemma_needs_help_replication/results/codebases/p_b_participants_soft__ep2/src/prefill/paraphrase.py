"""Paraphrasing of truncated responses (Appendix C.2).

To control for stylistic biases from using Gemma-generated text as prefills, all
truncations are paraphrased with Claude Sonnet, preserving meaning and emotion
level and keeping the same (possibly mid-sentence) ending point.
"""
from __future__ import annotations

from ..config import CFG
from ..llm import clients
from ..prompts.judge_prompts import PARAPHRASE_PROMPT


def paraphrase(text: str) -> str:
    if not text.strip():
        return text
    return clients.chat(
        CFG.paraphraser.provider, CFG.paraphraser.model,
        [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
        temperature=0.0, max_tokens=2048, disable_thinking=True,
    ).strip()
