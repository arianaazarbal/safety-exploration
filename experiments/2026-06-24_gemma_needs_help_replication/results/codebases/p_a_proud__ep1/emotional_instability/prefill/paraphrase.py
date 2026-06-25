"""Paraphrasing of truncated responses (Appendix C.2).

Truncations are paraphrased with Claude Sonnet to strip Gemma's stylistic
fingerprint while preserving meaning, tone and emotion level, so that base/instruct
differences reflect behaviour rather than mimicry of Gemma's phrasing.
"""

from __future__ import annotations

import dataclasses

from ..config import GENERATION, JUDGE_MODEL
from ..models import build_client
from ..models.base import Message
from ..prompts import PARAPHRASE_PROMPT
from .onset import PrefillItem

_PARA_GEN = dataclasses.replace(GENERATION, temperature=0.7, max_new_tokens=1024)


class Paraphraser:
    def __init__(self, spec=JUDGE_MODEL):
        self.client = build_client(spec)

    def paraphrase(self, text: str) -> str:
        if not text.strip():
            return text
        prompt = PARAPHRASE_PROMPT.format(text=text)
        return self.client.generate([Message("user", prompt)], gen=_PARA_GEN).text.strip()

    def paraphrase_item(self, item: PrefillItem) -> PrefillItem:
        return dataclasses.replace(item, prefill_text=self.paraphrase(item.raw_prefill_text))
