"""Paraphrasing of truncated prefills (Appendix C.2).

Uses Claude-Sonnet-4 to rewrite the truncated final assistant turn, preserving
meaning / tone / emotion level and the mid-sentence ending, to control for
Gemma's stylistic fingerprints when feeding the prefill to other models.
"""
from __future__ import annotations

from ..models.base import GenerationConfig, ModelClient
from ..prompts import paraphrase_prompt

_PARA_CFG = GenerationConfig(temperature=0.0, max_new_tokens=1024)


def paraphrase(client: ModelClient, text: str) -> str:
    out = client.chat([{"role": "user", "content": paraphrase_prompt(text)}], _PARA_CFG)
    return out.strip()
