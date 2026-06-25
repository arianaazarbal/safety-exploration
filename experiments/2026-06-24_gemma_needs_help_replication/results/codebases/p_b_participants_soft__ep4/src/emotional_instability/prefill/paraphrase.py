"""Paraphrasing of truncated prefills (Appendix C.2).

Used to control for stylistic biases from Gemma-generated text: every
truncation is paraphrased by Claude-Sonnet-4 before being used as a prefill, so
base/instruct models from other families aren't cued by Gemma's idiosyncratic
phrasing. The paraphrase preserves meaning, tone, and (critically) the
truncation point.
"""
from __future__ import annotations

from ..models import ChatMessage, GenerationConfig, ModelClient
from ..prompts.judge_prompts import PARAPHRASE_PROMPT


def paraphrase_text(paraphraser: ModelClient, text: str) -> str:
    if not text.strip():
        return text
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = paraphraser.generate(
        [ChatMessage("user", prompt)],
        GenerationConfig(temperature=0.7, max_new_tokens=1024, thinking=False),
    )
    return out.strip()
