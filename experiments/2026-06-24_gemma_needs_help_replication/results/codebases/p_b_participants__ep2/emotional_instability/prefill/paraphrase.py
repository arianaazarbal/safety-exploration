"""Paraphrasing of truncated prefills (Appendix C.2).

The truncations come from Gemma-generated text, which carries a recognisable
style. To stop a model from simply pattern-matching Gemma's voice, each prefill
is paraphrased by Claude Sonnet 4 — preserving meaning, tone, and the
mid-sentence cut-off — before being handed to the continuation models.
"""

from __future__ import annotations

from ..models.base import ChatMessage, ModelClient
from ..prompts.judge_prompts import PARAPHRASE


def paraphrase_prefill(paraphraser: ModelClient, text: str) -> str:
    prompt = PARAPHRASE.format(text=text)
    out = paraphraser.chat([ChatMessage("user", prompt)], n=1)[0].text
    return out.strip()
