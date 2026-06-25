"""Paraphrasing of truncated assistant turns (Section 3.1 / Appendix C.2).

To control for Gemma's stylistic fingerprint when prefilling *other* models'
generations, the truncated final assistant turn is paraphrased by Claude before
being used as a prefill. (In our Gemma-only scope this still matters: it keeps
the base vs instruct comparison from being biased by instruct-specific phrasing,
exactly as the paper intends.)
"""

from __future__ import annotations

from ..config import PARAPHRASE_MODEL
from ..models.llm_clients import AnthropicClient
from ..prompts import render_paraphrase_prompt


def paraphrase(text: str, client: AnthropicClient | None = None) -> str:
    if not text.strip():
        return text
    client = client or AnthropicClient(PARAPHRASE_MODEL)
    out = client.complete(render_paraphrase_prompt(text), max_tokens=1024,
                          temperature=0.0)
    return out.strip()
