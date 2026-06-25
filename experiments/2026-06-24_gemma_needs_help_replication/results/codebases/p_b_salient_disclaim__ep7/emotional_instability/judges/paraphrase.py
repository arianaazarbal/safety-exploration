"""Paraphrasing of truncated prefills (Appendix C.2).

Claude-Sonnet-4 rewrites the truncated assistant text to control for stylistic
biases from using Gemma-generated text, preserving meaning and emotion level
and keeping the truncation point roughly fixed.
"""

from __future__ import annotations

from typing import Optional

import config
from ..prompts import PARAPHRASE_PROMPT_TEMPLATE
from ._llm import anthropic_complete


def paraphrase_text(text: str, *, model: Optional[str] = None) -> str:
    model = model or config.PARAPHRASE_MODEL
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(text=text)
    out = anthropic_complete(model, prompt, max_tokens=1024, temperature=0.0)
    return out.strip()
