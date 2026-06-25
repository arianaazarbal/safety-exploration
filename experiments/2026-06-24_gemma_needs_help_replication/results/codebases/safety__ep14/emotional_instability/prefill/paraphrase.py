"""Paraphrasing of truncated prefills (Appendix C.2).

Rewrites Gemma-generated truncations to control for stylistic biases while
preserving meaning, tone, formality, and the (intentional) mid-sentence ending.
"""
from __future__ import annotations

from ..clients.base import GenerationConfig, ModelClient
from ..prompts import PARAPHRASE_PROMPT

PARAPHRASE_CFG = GenerationConfig(temperature=0.0, max_tokens=2048)


def paraphrase(client: ModelClient, text: str) -> str:
    if not text.strip():
        return text
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = client.chat([{"role": "user", "content": prompt}], PARAPHRASE_CFG)
    return out.strip()
