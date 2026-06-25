"""Paraphrasing of truncated prefills (Appendix C.2).

Controls for stylistic biases from Gemma-generated text by rewriting the
truncation while preserving meaning, tone, and the (mid-sentence) ending point.
"""

from __future__ import annotations

from .. import prompts
from ..llm_client import anthropic_complete

import config


def paraphrase(text: str, model: str = config.PARAPHRASE_MODEL) -> str:
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    out = anthropic_complete(model, prompt, max_tokens=1024, temperature=0.7)
    return out.strip()
